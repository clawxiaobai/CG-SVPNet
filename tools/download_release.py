"""Download, verify and safely extract the pinned public reproduction assets."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def extract_checked(archive, destination):
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            rel = PurePosixPath(item.filename)
            if (rel.is_absolute() or '..' in rel.parts or '\\' in item.filename
                    or ':' in item.filename or item.is_dir()):
                raise ValueError(f'Unexpected archive path: {item.filename}')
            target = destination.joinpath(*rel.parts)
            base = destination.resolve()
            if not target.resolve().is_relative_to(base):
                raise ValueError(f'Archive path leaves destination: {item.filename}')
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if target.stat().st_size != item.file_size:
                    raise FileExistsError(f'Existing file differs: {target}')
                with z.open(item) as src:
                    packed_hash = hashlib.file_digest(src, 'sha256').hexdigest()
                if sha256(target) != packed_hash:
                    raise FileExistsError(f'Existing file differs: {target}')
                continue
            temporary = target.with_name(target.name + '.download.tmp')
            with z.open(item) as src, temporary.open('wb') as dst:
                shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
            temporary.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', choices=['weights', 'data', 'examples', 'all'], default='weights')
    parser.add_argument('--destination', type=Path, default=ROOT)
    parser.add_argument('--downloads', type=Path, default=ROOT / 'downloads')
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'release_manifest.json').read_text(encoding='utf-8'))
    selected = [a for a in manifest['assets'] if args.scope in ('all', a['scope'])]
    if not selected:
        raise ValueError('No assets match the selected scope')
    args.downloads.mkdir(parents=True, exist_ok=True)
    args.destination.mkdir(parents=True, exist_ok=True)
    for i, asset in enumerate(selected, 1):
        path = args.downloads / asset['name']
        valid = path.is_file() and path.stat().st_size == asset['bytes'] and sha256(path) == asset['sha256']
        if not valid:
            if args.offline:
                raise FileNotFoundError(f'Missing or invalid offline archive: {path}')
            temporary = path.with_suffix(path.suffix + '.tmp')
            print(f'[{i}/{len(selected)}] Downloading {asset["name"]} ({asset["bytes"]:,} bytes)', flush=True)
            request = urllib.request.Request(asset['url'], headers={'User-Agent': 'CG-SVPNet-reproduction'})
            digest = hashlib.sha256()
            with urllib.request.urlopen(request, timeout=120) as src, temporary.open('wb') as dst:
                while block := src.read(8 * 1024 * 1024):
                    digest.update(block)
                    dst.write(block)
            if temporary.stat().st_size != asset['bytes'] or digest.hexdigest() != asset['sha256']:
                raise ValueError(f'Download checksum mismatch: {asset["name"]}')
            temporary.replace(path)
        print(f'[{i}/{len(selected)}] SHA-256 verified; extracting {asset["name"]}', flush=True)
        extract_checked(path, args.destination)
    print('Selected assets verified and extracted. Use the application full-hash check to verify the complete dataset.')


if __name__ == '__main__':
    main()
