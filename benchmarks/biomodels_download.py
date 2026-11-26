"""Download the curated BioModels (SBML only) into tests/models/biomodels, for benchmarks/biomodels_compare.py.

    python benchmarks/biomodels_download.py            # BIOMD0000000001 upward, until 3 empty batches
    python benchmarks/biomodels_download.py --last 1300

Uses BioModels' bulk download (at most 100 models per request), as libroadrunner's test/python/download_biomodels.py
does, without the bioservices dependency. Missing ids and non-SBML files (CellML) are skipped. Already downloaded
models are not fetched again. The folder is not committed (see .gitignore).
"""
import argparse
import io
import os
import time
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'tests', 'models', 'biomodels')
URL = 'https://www.ebi.ac.uk/biomodels/search/download?models='


def fetch(ids, tries=3):
    for k in range(tries):
        try:
            with urllib.request.urlopen(URL + ','.join(ids), timeout=300) as r:
                return zipfile.ZipFile(io.BytesIO(r.read()))
        except Exception as e:  # network hiccups: retry, then give up on this batch
            if k == tries - 1:
                print('  batch %s..%s failed: %s' % (ids[0], ids[-1], e))
                return None
            time.sleep(5 * (k + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--first', type=int, default=1)
    ap.add_argument('--last', type=int, default=None, help='last model number (default: until 3 empty batches)')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    n, empty, saved = a.first, 0, 0
    while (a.last is None and empty < 3) or (a.last is not None and n <= a.last):
        end = n + 99 if a.last is None else min(n + 99, a.last)
        ids = ['BIOMD%010d' % i for i in range(n, end + 1)
               if not os.path.exists(os.path.join(OUT, 'BIOMD%010d.xml' % i))]
        got = 0
        if ids:
            z = fetch(ids)
            for info in (z.infolist() if z else []):
                name = os.path.basename(info.filename)
                if name.endswith('.xml'):
                    with open(os.path.join(OUT, name), 'wb') as f:
                        f.write(z.read(info))
                    got += 1
            saved += got
        have = len([f for f in os.listdir(OUT) if f.endswith('.xml')])
        print('BIOMD%010d..%010d: %d new SBML files (%d in total)' % (n, end, got, have), flush=True)
        empty = empty + 1 if (ids and got == 0) else 0
        n = end + 1
    print('done: %d new, folder %s' % (saved, os.path.abspath(OUT)))


if __name__ == '__main__':
    main()
