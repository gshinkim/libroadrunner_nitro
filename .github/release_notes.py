"""Release description for a GitHub Release of roadrunner-nitro: what it is and a pip install command per wheel.

    python .github/release_notes.py <owner/repo> <tag> <folder with the wheels>   -> markdown on stdout

Run by .github/workflows/wheels.yml for each v* tag; the commands link to that release's own wheel files.
"""
import os
import sys

# wheel platform tag (a substring of the file name) -> what a tester calls the platform
PLATFORMS = [
    ('win_amd64', 'Windows (x64)'),
    ('macosx_11_0_arm64', 'macOS (Apple Silicon)'),
    ('macosx_10_12_x86_64', 'macOS (Intel)'),
    ('manylinux_2_28_x86_64', 'Linux (x86_64)'),
    ('manylinux_2_28_aarch64', 'Linux (ARM64)'),
]


def main(repo, tag, folder):
    wheels = sorted(f for f in os.listdir(folder) if f.endswith('.whl'))
    base = 'https://github.com/%s/releases/download/%s/' % (repo, tag)
    version = tag.lstrip('v')
    candidate = any(x in version for x in ('rc', 'a', 'b'))
    out = ['## roadrunner-nitro %s' % version, '',
           'A fast drop-in replacement for [libroadrunner](https://github.com/sys-bio/roadrunner): the same Python API '
           'and results, written in Rust. Use it as `import roadrunner_nitro as roadrunner`.', '']
    if candidate:
        out += ['**Release candidate for testing.** Please report problems as issues on this repository.', '']
    out += ['### Install', '',
            'Needs Python 3.10 or newer; no Rust or compiler. Pick the line for your computer:', '',
            '| Platform | Command |', '| --- | --- |']
    for key, label in PLATFORMS:
        for w in wheels:
            if key in w:
                out.append('| %s | `pip install %s%s` |' % (label, base, w))
    rest = [w for w in wheels if not any(k in w for k, _ in PLATFORMS)]
    for w in rest:
        out.append('| other | `pip install %s%s` |' % (base, w))
    out += ['', 'Check it: `python -c "import roadrunner_nitro as rn; print(rn.__version__)"`', '',
            'The wheels and the source package are also listed under **Assets** below.']
    print('\n'.join(out))


if __name__ == '__main__':
    main(*sys.argv[1:4])
