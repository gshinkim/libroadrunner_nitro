"""NamedArray: libroadrunner's numpy array with row and column names."""
# Moved from Tellurium Nitro (tellurium_nitro/roadrunner/extended_roadrunner.py, Apache 2.0).
import numpy as np


class NamedArray(np.ndarray):
    """numpy array with ``rownames`` / ``colnames``, printed as libroadrunner prints its NamedArray.
    ``a['S1']``: the column (or row) of that name."""

    def __new__(cls, a, rownames=(), colnames=()):
        """NamedArray(shape) as libroadrunner (a float array of that shape), or NamedArray(data, rownames,
        colnames)."""
        if isinstance(a, int) or (isinstance(a, tuple) and a and all(isinstance(x, int) for x in a)):
            obj = np.zeros(a).view(cls)
        else:
            obj = np.asarray(a).view(cls)
        obj.rownames, obj.colnames = list(rownames), list(colnames)
        return obj

    def __array_finalize__(self, obj):
        self.rownames = list(getattr(obj, 'rownames', ()))
        self.colnames = list(getattr(obj, 'colnames', ()))

    # Pickling as libroadrunner's NamedArray: NamedArray(shape) then __setstate__(dict).
    def __getstate__(self):
        a = np.ascontiguousarray(self, dtype=float)
        return {'_pickle_version': 1, 'array': a.tobytes(), 'rownames': list(self.rownames),
                'colnames': list(self.colnames), 'nDims': a.ndim,
                'dim1': a.shape[0] if a.ndim > 0 else 0, 'dim2': a.shape[1] if a.ndim > 1 else 0}

    def __reduce_ex__(self, protocol):
        return NamedArray, (tuple(self.shape),), self.__getstate__(), None, None

    def __reduce__(self):
        return self.__reduce_ex__(2)

    def __setstate__(self, state):
        if isinstance(state, dict):
            shape = (state['dim1'], state['dim2'])[:state['nDims']]
            np.ndarray.__setstate__(self, (1, shape, np.dtype(float), False, state['array']))
            self.rownames, self.colnames = list(state['rownames']), list(state['colnames'])
        else:  # pickles from roadrunner_nitro 0.1: (ndarray state, rownames, colnames)
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
