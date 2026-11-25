"""NamedArray: libroadrunner's numpy array with row and column names."""
# Moved from Tellurium Nitro (tellurium_nitro/roadrunner/extended_roadrunner.py, Apache 2.0).
import numpy as np


class NamedArray(np.ndarray):
    """numpy array with ``rownames`` / ``colnames``, printed as libroadrunner prints its NamedArray.
    ``a['S1']``: the column (or row) of that name."""

    def __new__(cls, a, rownames=(), colnames=()):
        obj = np.asarray(a).view(cls)
        obj.rownames, obj.colnames = list(rownames), list(colnames)
        return obj

    def __array_finalize__(self, obj):
        self.rownames = list(getattr(obj, 'rownames', ()))
        self.colnames = list(getattr(obj, 'colnames', ()))

    def __reduce__(self):
        f, args, state = super(NamedArray, self).__reduce__()
        return f, args, (state, self.rownames, self.colnames)

    def __setstate__(self, state):
        super(NamedArray, self).__setstate__(state[0])
        self.rownames, self.colnames = state[1], state[2]

    def __getitem__(self, key):
        if isinstance(key, str):
            a = np.asarray(self)
            if key in self.colnames:
                return a[..., self.colnames.index(key)]
            if key in self.rownames:
                return a[self.rownames.index(key)]
            raise KeyError(key)
        return super(NamedArray, self).__getitem__(key)

    def __str__(self):
        a = np.asarray(self)
        if a.ndim != 2 or not len(a) or len(self.colnames) != a.shape[1] or \
                (self.rownames and len(self.rownames) != a.shape[0]):
            return str(a)
        cells = [[format(v, 'g') for v in row] for row in a]
        w = [max([len(n)] + [len(r[j]) for r in cells]) + 1 for j, n in enumerate(self.colnames)]
        rn = max([len(n) for n in self.rownames] or [0])
        out = [' ' * rn + '   ' + ','.join(n.rjust(x) for n, x in zip(self.colnames, w))]
        for i, r in enumerate(cells):
            row = ','.join(c.rjust(x) for c, x in zip(r, w))
            lead = self.rownames[i].ljust(rn) if rn else ''
            out.append(lead + (' [[' if i == 0 else '  [') + row + (']]' if i == len(cells) - 1 else '],'))
        return '\n'.join(out) + '\n'

    __repr__ = __str__


def _named(t):
    """(array, row ids, column ids) from roadrunner_nitro.core -> NamedArray."""
    return NamedArray(t[0], t[1], t[2])
