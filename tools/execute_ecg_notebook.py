from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "notebooks" / "05_ecg_model.ipynb"
notebook = nbformat.read(path, as_version=4)
client = NotebookClient(
    notebook,
    timeout=3600,
    kernel_name="cardioscope",
    resources={"metadata": {"path": str(path.parent)}},
)
client.execute()
nbformat.write(notebook, path)
print("Saved executed 05_ecg_model.ipynb", flush=True)
