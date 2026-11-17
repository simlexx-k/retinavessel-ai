# Research provenance

This directory is the durable evidence ledger for completed RetinaVessel
experiments. It keeps executed notebooks and compact result materials in Git,
while large checkpoint archives remain external and are identified by size,
SHA-256 and expected contents in `manifest.json`.

## Layout

- `notebooks/executed/`: immutable Kaggle notebooks with cell outputs.
- `runs/<seed>/`: exact training manifests and compact validation tables
  extracted from the verified run archives.
- `manifest.json`: machine-readable notebook, dataset, run, checkpoint and
  external-artifact identities.
- `artifact-checksums.sha256`: conventional checksums for the external ZIPs.
- `verify.py`: verifies tracked notebooks and, optionally, downloaded ZIPs.

Notebook templates normally remain at the repository root, but Kaggle's GitHub
save operation replaced the root V5 file with the byte-exact executed download.
The manifest records that file as `original_path`; normalized archival copies
of completed V1–V5 notebooks live under `notebooks/executed/`. The deterministic
three-seed aggregation inputs, outputs and script are under `aggregation/` and
`aggregate.py`. The root V6 notebook is the planned, not-yet-executed official
test stage; its checksum is recorded under `planned_notebooks` in the manifest.

## Verify

Verify the Git-tracked evidence:

```bash
python provenance/verify.py
```

Also verify the downloaded checkpoint/result archives:

```bash
python provenance/verify.py \
  --artifacts-dir /mnt/c/Users/simon/Downloads \
  --require-artifacts
```

## Adding a completed run

1. Save the completed Kaggle notebook version and download both the executed
   `.ipynb` and final artifact ZIP.
2. Run a full ZIP integrity test and calculate SHA-256 before extracting data.
3. Copy the executed notebook into `notebooks/executed/`; do not clear or edit
   its outputs.
4. Copy only compact manifests and summary tables into `runs/<seed>/`.
5. Add the notebook, archive, checkpoint hashes and protocol facts to
   `manifest.json`, then update `artifact-checksums.sha256`.
6. Run `verify.py` and the repository test suite before committing.

The official FIVES test partition remains sealed. Provenance records must not
claim official-test evaluation until the multi-seed decision has been frozen
and the single predeclared final-test stage is executed.
