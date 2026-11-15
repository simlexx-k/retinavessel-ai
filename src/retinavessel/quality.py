"""Read the FIVES quality-assessment workbook without an Excel dependency."""
from dataclasses import dataclass
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


@dataclass(frozen=True)
class QualityScore:
    illumination_color: int
    blur: int
    low_contrast: int

    @property
    def all_good(self):
        return self.illumination_color == self.blur == self.low_contrast == 1

    @property
    def values(self):
        return (self.illumination_color, self.blur, self.low_contrast)


def _cell_value(cell, shared_strings):
    value = cell.find(f"{{{MAIN_NS}}}v")
    if value is None or value.text is None:
        return ""
    raw = value.text
    return shared_strings[int(raw)] if cell.get("t") == "s" else raw


def _column_name(reference):
    match = re.match(r"[A-Z]+", reference or "")
    if not match:
        raise ValueError(f"Invalid workbook cell reference: {reference!r}")
    return match.group(0)


def read_quality_workbook(path):
    """Return ``{'train': {sample_id: QualityScore}, 'test': ...}``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"FIVES quality workbook not found: {path}")

    ns = {"m": MAIN_NS, "r": REL_NS, "p": PKG_REL_NS}
    with ZipFile(path) as archive:
        shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        shared = ["".join(item.itertext())
                  for item in shared_root.findall("m:si", ns)]

        relationships = ET.fromstring(
            archive.read("xl/_rels/workbook.xml.rels")
        )
        targets = {
            rel.get("Id"): rel.get("Target")
            for rel in relationships.findall("p:Relationship", ns)
        }
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        sheet_paths = {}
        for sheet in workbook.findall("m:sheets/m:sheet", ns):
            name = sheet.get("name", "").casefold()
            relationship_id = sheet.get(f"{{{REL_NS}}}id")
            target = targets.get(relationship_id)
            if target:
                sheet_paths[name] = "xl/" + target.lstrip("/")

        result = {}
        expected_headers = {
            "A": "Disease", "B": "Number", "C": "IC",
            "D": "Blur", "E": "LC",
        }
        for split in ("train", "test"):
            if split not in sheet_paths:
                raise ValueError(f"Quality workbook has no {split.title()} sheet")
            root = ET.fromstring(archive.read(sheet_paths[split]))
            rows = root.findall(".//m:sheetData/m:row", ns)
            if not rows:
                raise ValueError(f"Quality workbook {split.title()} sheet is empty")

            def cells_by_column(row):
                return {
                    _column_name(cell.get("r")): _cell_value(cell, shared)
                    for cell in row.findall("m:c", ns)
                }

            headers = cells_by_column(rows[0])
            if any(headers.get(col) != label
                   for col, label in expected_headers.items()):
                raise ValueError(
                    f"Unexpected headers in {split.title()} quality sheet: {headers}"
                )

            scores = {}
            for row in rows[1:]:
                cells = cells_by_column(row)
                condition = cells.get("A", "").upper()
                if condition not in "ADGN":
                    raise ValueError(f"Invalid FIVES disease code: {condition!r}")
                number = int(float(cells["B"]))
                values = tuple(int(cells[col]) for col in ("C", "D", "E"))
                if any(value not in (0, 1) for value in values):
                    raise ValueError(f"Invalid quality scores for {number}_{condition}")
                sample_id = f"{number}_{condition}"
                if sample_id in scores:
                    raise ValueError(f"Duplicate quality row for {split}/{sample_id}")
                scores[sample_id] = QualityScore(*values)
            result[split] = scores

    return result
