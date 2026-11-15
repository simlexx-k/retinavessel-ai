"""Verified public FIVES Figshare download; optional extraction using external 7z."""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import requests

URL='https://ndownloader.figshare.com/files/34969398'
EXPECTED_MD5='789c80dd5376a82063e27fa49192bac9'


def download(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        print(f'Downloading official FIVES (approximately 1.76 GB) to {path}')
        with requests.get(URL,stream=True,timeout=60) as response:
            response.raise_for_status()
            with path.open('wb') as f:
                for chunk in response.iter_content(chunk_size=1024*1024):
                    if chunk:f.write(chunk)
    md5=hashlib.md5()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):md5.update(block)
    if md5.hexdigest()!=EXPECTED_MD5:
        raise RuntimeError(f'Checksum mismatch: {md5.hexdigest()} != {EXPECTED_MD5}. Remove and redownload file.')
    print('Official archive checksum verified.')
    return path


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='data/fives_original.rar')
    p.add_argument('--extract',action='store_true');p.add_argument('--dest',default='data/fives')
    args=p.parse_args();archive=download(args.out)
    if args.extract:
        cmd=shutil.which('7z') or shutil.which('7za') or shutil.which('7zz')
        if not cmd:
            raise RuntimeError('7-Zip executable not found. Install 7z or extract the RAR manually.')
        Path(args.dest).mkdir(parents=True,exist_ok=True)
        subprocess.run([cmd,'x',str(archive),f'-o{args.dest}','-y'],check=True)
        print('Extracted to:', args.dest)
if __name__=='__main__':main()
