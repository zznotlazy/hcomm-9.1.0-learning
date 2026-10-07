"""Read-only verification of upstream source content and cloud learning assets."""
from pathlib import Path
import hashlib
import json
import re

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
manifest = json.loads((OUT / 'source_manifest.json').read_text(encoding='utf-8'))
provenance = json.loads((OUT / 'source_provenance.json').read_text(encoding='utf-8'))
exceptions = set(provenance['allowed_upstream_changes'])
issues = []
images = 0
for relative, expected in manifest.items():
    path = ROOT / relative
    if not path.is_file():
        issues.append(f'Missing upstream file: {relative}')
        continue
    data = path.read_bytes()
    if relative not in exceptions and (
        hashlib.sha256(data).hexdigest() != expected['sha256'] or len(data) != expected['size']
    ):
        issues.append(f'Changed upstream content: {relative}')
    if path.suffix == '.png':
        images += 1
        if not data.startswith(b'\x89PNG\r\n\x1a\n'):
            issues.append(f'PNG missing or LFS pointer: {relative}')
if images != 92 or len(manifest) != provenance['tracked_files']:
    issues.append('Snapshot file/image count changed')
if 'filter=lfs' in (ROOT / '.gitattributes').read_text(encoding='utf-8'):
    issues.append('Learning snapshot unexpectedly requires LFS')
for relative in ['AGENTS.md', 'README.md']:
    if not (ROOT / relative).is_file():
        issues.append(f'Missing entrypoint: {relative}')
for name in ['CLOUD_LEARNING.md', 'baseline.json', 'selection_manifest.json', 'validation.json']:
    if not (OUT / name).is_file():
        issues.append(f'Missing learning artifact: {name}')
for path in OUT.rglob('*'):
    if path.suffix not in {'.md', '.json'}:
        continue
    text = path.read_text(encoding='utf-8')
    if re.search(r'[A-Za-z]:[/\\](?:Desktop|Users)|D:/Desktop|C:/Users', text):
        issues.append(f'Personal/local path retained: {path.relative_to(ROOT)}')
for path in [ROOT / 'README.md', ROOT / 'AGENTS.md', OUT / 'CLOUD_LEARNING.md']:
    text = re.sub(r'```[^\n]*\n[\s\S]*?```', '', path.read_text(encoding='utf-8'))
    for match in re.finditer(r'\]\(([^\n)]+)\)', text):
        destination = match[1]
        if destination.startswith(('https:', 'http:', '#')):
            continue
        if not (path.parent / destination.split('#')[0]).is_file():
            issues.append(f'Missing entry link: {destination}')
validation = json.loads((OUT / 'validation.json').read_text(encoding='utf-8'))
if validation.get('issues') or validation.get('registered_commands') != 15:
    issues.append('Evidence validation failed or command coverage changed')
print(json.dumps({'upstream_files':len(manifest), 'png_assets':images,
                  'intentional_changes':sorted(exceptions), 'issues':issues,
                  'validation':'Source snapshot and teaching assets only; HCCL not executed'},
                 ensure_ascii=False))
if issues:
    raise SystemExit(1)
