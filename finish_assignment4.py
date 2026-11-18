import os
from pathlib import Path
import subprocess
import sys

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent

if __name__ == '__main__':
    os.chdir(ROOT)
    subprocess.run([sys.executable, 'train_advanced.py', *sys.argv[1:]], check=True)
    path = ROOT / 'Assignment4_Eric_Liu.ipynb'
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(notebook, timeout=900, kernel_name='python3')
    # Use the same Python installation for training and the notebook.
    client.km = client.create_kernel_manager()
    client.km.kernel_spec.argv[0] = sys.executable
    client.execute()
    tmp = path.with_suffix('.tmp')
    nbformat.write(notebook, tmp)
    tmp.replace(path)
    print('Saved', path)
