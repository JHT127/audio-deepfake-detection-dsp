# Dataset

**Source:** ASVspoof2019 Logical Access (LA), eval partition.
Edinburgh DataShare: https://doi.org/10.7488/ds/2555

**Subset:** 150 bonafide + 150 spoof files, selected from the official
evaluation protocol file via `scripts/select_subset.py`.

**Reproduction:** Files are not committed to this repo due to size.
To reproduce the exact subset:

1. `python scripts/list_remote_contents.py`
2. `python scripts/fetch_protocol.py`
3. `python scripts/select_subset.py`
4. `python scripts/extract_subset.py`
5. `python scripts/verify_subset.py`

This downloads only the selected files (~15-30 MB) via HTTP range
requests, not the full 7.12 GB archive.
