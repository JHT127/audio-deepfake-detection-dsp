# scripts/list_remote_contents.py
"""
Step 1: Connect to the remote LA.zip and list its contents
without downloading the archive. Run this first to confirm
the internal folder structure and find the protocol file path.
"""
from remotezip import RemoteZip

URL = "https://datashare.ed.ac.uk/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269/download"

with RemoteZip(URL) as zf:
    names = zf.namelist()
    print(f"Total entries: {len(names)}")
    print("\nSample entries:")
    for n in names[:15]:
        print(n)

    print("\nProtocol-related files:")
    for n in names:
        if "protocol" in n.lower():
            print(n)