"""Refresh the source archive embedded in the Kaggle notebook (standard library only)."""

import base64
import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    notebook_path = root / 'kaggle_train_relationformer.ipynb'
    notebook = json.loads(notebook_path.read_text())
    cells = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code'
             and any(line.startswith('SOURCE_ARCHIVE_B64 = ') for line in cell['source'])]
    if len(cells) != 1:
        raise ValueError('Expected exactly one source archive cell in the notebook')
    files = sorted(set(list(root.glob('*.py')) + list((root / 'models').rglob('*.py'))
                       + list((root / 'metric_topo').rglob('*.py'))
                       + [root / 'configs/road_2D.yaml', root / 'requirements-kaggle.txt']))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            # Fixed timestamps make unchanged inputs produce the same archive.
            info = zipfile.ZipInfo(path.relative_to(root).as_posix(), date_time=(2026, 9, 10, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    snapshot = buffer.getvalue()
    values = {
        'SOURCE_BASE_COMMIT': subprocess.check_output(
            ['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
        'SOURCE_SNAPSHOT_SHA256': hashlib.sha256(snapshot).hexdigest(),
        'SOURCE_ARCHIVE_B64': base64.b64encode(snapshot).decode(),
    }
    cell = cells[0]
    for index, line in enumerate(cell['source']):
        for name, value in values.items():
            if line.startswith(name + ' = '):
                cell['source'][index] = f'{name} = {value!r}\n'
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            cell.update(execution_count=None, outputs=[])
    notebook_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n')
    print(f"Embedded {len(files)} files; SHA-256: {values['SOURCE_SNAPSHOT_SHA256']}")


if __name__ == '__main__':
    main()
