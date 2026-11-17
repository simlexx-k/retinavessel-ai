"""Verify tracked provenance records and optional external run archives."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
from zipfile import BadZipFile, ZipFile


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = Path(__file__).with_name("manifest.json")


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("--artifacts-dir", type=Path)
    parser.add_argument("--require-artifacts", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST_PATH.read_text())
    failures: list[str] = []

    if manifest["dataset"]["official_test_evaluated"] is not False:
        failures.append("manifest no longer records the official test as sealed")

    for record in manifest["executed_notebooks"]:
        path = ROOT / record["path"]
        if not path.is_file():
            failures.append(f"missing notebook: {record['path']}")
            continue
        if digest(path) != record["repository_sha256"]:
            failures.append(f"notebook checksum mismatch: {record['path']}")
        notebook = json.loads(path.read_text())
        code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        executed = sum(cell.get("execution_count") is not None for cell in code)
        errors = sum(
            output.get("output_type") == "error"
            for cell in code
            for output in cell.get("outputs", [])
        )
        if len(code) != record["code_cells"]:
            failures.append(f"code-cell count mismatch: {record['path']}")
        if executed != record["executed_code_cells"]:
            failures.append(f"execution count mismatch: {record['path']}")
        if errors != record["error_outputs"]:
            failures.append(f"error output found: {record['path']}")
        if "source_template" in record:
            source = ROOT / record["source_template"]
            if not source.is_file() or digest(source) != record["source_template_sha256"]:
                failures.append(f"source template mismatch: {record['source_template']}")
        if "original_path" in record:
            original = ROOT / record["original_path"]
            if not original.is_file() or digest(original) != record["original_sha256"]:
                failures.append(f"original notebook mismatch: {record['original_path']}")
        print(f"OK notebook {record['path']} ({executed}/{len(code)} cells executed)")

    for record in manifest["planned_notebooks"]:
        path = ROOT / record["path"]
        if not path.is_file() or digest(path) != record["sha256"]:
            failures.append(f"planned notebook mismatch: {record['path']}")
        else:
            print(f"OK planned notebook {record['path']}")

    for record in manifest["tracked_materials"]:
        path = ROOT / record["path"]
        if not path.is_file() or digest(path) != record["sha256"]:
            failures.append(f"tracked material mismatch: {record['path']}")
        else:
            print(f"OK material {record['path']}")

    for record in manifest["external_artifacts"]:
        if record["official_test_evaluated"] is not False:
            failures.append(f"official-test boundary violated: {record['run_id']}")
        path = args.artifacts_dir / record["filename"] if args.artifacts_dir else None
        if path is None or not path.is_file():
            message = f"external artifact unavailable: {record['filename']}"
            if args.require_artifacts:
                failures.append(message)
            else:
                print(f"SKIP {message}")
            continue
        if path.stat().st_size != record["bytes"]:
            failures.append(f"artifact size mismatch: {record['filename']}")
            continue
        if digest(path) != record["sha256"]:
            failures.append(f"artifact checksum mismatch: {record['filename']}")
            continue
        try:
            with ZipFile(path) as archive:
                corrupt = archive.testzip()
        except BadZipFile:
            corrupt = "invalid ZIP"
        if corrupt is not None:
            failures.append(f"artifact ZIP failure ({corrupt}): {record['filename']}")
        else:
            print(f"OK artifact {record['filename']}")

    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        return 1
    print("Provenance verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
