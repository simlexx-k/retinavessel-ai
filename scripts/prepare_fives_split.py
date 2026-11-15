"""Freeze the audited FIVES training/validation split before model fitting."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from retinavessel.data import (
    find_fives_root,
    load_manifest,
    quality_stratified_train_val,
)
from retinavessel.quality import read_quality_workbook


ARCHIVE_MD5 = "789c80dd5376a82063e27fa49192bac9"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric_id(sample_id):
    return int(sample_id.split("_", 1)[0])


def split_counts(samples, quality_scores):
    counts = Counter(
        (sample.condition,
         "all_good" if quality_scores[sample.id].all_good else "degraded")
        for sample in samples
    )
    return {
        f"{condition}:{quality_class}": count
        for (condition, quality_class), count in sorted(counts.items())
    }


def prepare(data, audit_path, out, seed=42, val_fraction=0.15):
    root = find_fives_root(data)
    official_train = load_manifest(root, "train")
    official_test = load_manifest(root, "test")
    audit_path = Path(audit_path)
    audit = json.loads(audit_path.read_text())
    summary = audit["summary"]

    if summary["total_paired_images"] != 800 or summary["train"] != 600 or summary["test"] != 200:
        raise ValueError("Audit does not describe the expected 600/200 FIVES release")
    if summary["missing_or_mismatched_sizes"]:
        raise ValueError("Cannot freeze a split while image/mask sizes are mismatched")
    if summary["identical_pixels_cross_split"]:
        raise ValueError("Cross-split duplicates require a protocol amendment")

    workbook = root / "Quality Assessment.xlsx"
    quality = read_quality_workbook(workbook)
    train_ids = {sample.id for sample in official_train}
    test_ids = {sample.id for sample in official_test}
    if set(quality["train"]) != train_ids or set(quality["test"]) != test_ids:
        raise ValueError("Quality workbook rows do not exactly match FIVES filenames")

    excluded = {}
    for sample in audit["samples"]:
        if sample["split"] == "train" and sample["all_background_mask"]:
            excluded[sample["id"]] = (
                "All-background vessel mask confirmed by automated audit and visual review; "
                "retained in raw data for quality/failure analysis but excluded from supervised fitting."
            )

    train_duplicate_groups = []
    test_duplicate_groups = []
    for group in summary["identical_pixel_groups"]:
        group_ids = [entry[1] for entry in group]
        splits = {entry[0] for entry in group}
        if splits == {"train"}:
            retained = min(group_ids, key=lambda value: (numeric_id(value), value))
            removed = sorted(set(group_ids) - {retained}, key=numeric_id)
            for sample_id in removed:
                excluded[sample_id] = (
                    f"Exact pixel duplicate of {retained}; deterministic policy retains the "
                    "lowest numeric ID to avoid image/annotation-informed selection."
                )
            train_duplicate_groups.append({"retained": retained, "excluded": removed})
        elif splits == {"test"}:
            retained = min(group_ids, key=lambda value: (numeric_id(value), value))
            test_duplicate_groups.append({
                "official_primary": sorted(group_ids, key=numeric_id),
                "deduplicated_sensitivity_representative": retained,
            })
        else:
            raise ValueError(f"Unexpected duplicate group spanning splits: {group}")

    training, validation = quality_stratified_train_val(
        official_train,
        quality["train"],
        seed=seed,
        val_fraction=val_fraction,
        excluded_ids=excluded,
    )
    eligible_ids = train_ids - set(excluded)
    assigned_ids = {sample.id for sample in training + validation}
    if assigned_ids != eligible_ids:
        raise AssertionError("Frozen split does not cover every eligible training image exactly once")

    manifest = {
        "schema_version": 1,
        "dataset": "FIVES original Figshare release",
        "archive_md5": ARCHIVE_MD5,
        "audit_report": str(audit_path),
        "audit_report_sha256": file_sha256(audit_path),
        "quality_workbook_sha256": file_sha256(workbook),
        "seed": seed,
        "validation_fraction_requested": val_fraction,
        "stratification": "disease code x official quality class (all fields good vs any field poor)",
        "quality_coding": {
            "IC": "illumination and color distortion; 1=good, 0=poor",
            "Blur": "1=good, 0=poor",
            "LC": "low contrast; 1=good, 0=poor",
        },
        "counts": {
            "official_train": len(official_train),
            "excluded": len(excluded),
            "eligible": len(eligible_ids),
            "train": len(training),
            "validation": len(validation),
            "official_test_untouched": len(official_test),
        },
        "train_strata": split_counts(training, quality["train"]),
        "validation_strata": split_counts(validation, quality["train"]),
        "excluded": dict(sorted(excluded.items(), key=lambda item: numeric_id(item[0]))),
        "train_duplicate_policy": {
            "rule": "For exact-pixel duplicates, retain the lowest numeric ID and exclude later copies.",
            "groups": train_duplicate_groups,
        },
        "official_test_policy": {
            "primary": "Retain all 200 official test records and never use them for model selection.",
            "sensitivity": "Also report a secondary result with one representative per exact-duplicate group.",
            "duplicate_groups": test_duplicate_groups,
        },
        "limitations": [
            "FIVES does not publish image-to-participant identifiers, so participant-disjoint splitting cannot be verified.",
            "The two excluded all-background records remain part of the immutable raw dataset and quality stress analysis.",
        ],
        "train": [sample.id for sample in training],
        "validation": [sample.id for sample in validation],
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "out": str(out),
        "counts": manifest["counts"],
        "train_strata": manifest["train_strata"],
        "validation_strata": manifest["validation_strata"],
        "excluded_ids": list(manifest["excluded"]),
    }, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--audit", default="outputs/data_audit.json")
    parser.add_argument("--out", default="splits/fives_seed42.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    args = parser.parse_args()
    prepare(args.data, args.audit, args.out, args.seed, args.val_fraction)


if __name__ == "__main__":
    main()
