import sys, os, time, nbformat
from nbclient import NotebookClient
order = sys.argv[1:]
for p in order:
    t=time.time(); nb = nbformat.read(p, as_version=4)
    NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": os.path.dirname(p)}}).execute()
    nbformat.write(nb, p); print(f"OK {p} ({time.time()-t:.0f}s)")
