# scripts/fetch_protocol.py
"""
Step 2: Download only the protocol file (a few hundred KB)
from the remote zip, so we can select our subset before
touching any audio.
"""
import os
from remotezip import RemoteZip

URL = "https://datashare.ed.ac.uk/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269/download"
PROTOCOL_PATH_IN_ZIP = "LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"  # from step 1 output
OUT_DIR = "data/external"

os.makedirs(OUT_DIR, exist_ok=True)

with RemoteZip(URL) as zf:
    zf.extract(PROTOCOL_PATH_IN_ZIP, "data/_tmp_proto")

# flatten to data/external/protocol.txt
import shutil
src = os.path.join("data/_tmp_proto", PROTOCOL_PATH_IN_ZIP)
dst = os.path.join(OUT_DIR, "protocol.txt")
shutil.move(src, dst)
shutil.rmtree("data/_tmp_proto")
print(f"Saved protocol to {dst}")