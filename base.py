#!/usr/bin/env python
#! nix-shell -i python -p "python38.withPackages(p:[p.pillow])"

import sys
from PIL import Image
import numpy as np
import scipy.ndimage.filters


class Img:
    def __init__(self, fname, zoom):
        self._img = Image.open(fname)
        self._pixels = self._img.load()
        self._zoom = zoom

        self.size = self._img.size[0] // zoom, self._img.size[1] // zoom

    def __getitem__(self, xy):
        xy = xy[0] * self._zoom, xy[1] * self._zoom
        try:
            c = self._pixels[xy]
        except IndexError:
            return False
        return c[0] + c[1] + c[2] > 382

    def to_numpy(self):
        im = np.zeros(self.size)
        for y in range(self.size[1]):
            for x in range(self.size[0]):
                im[x][y] = self[x, y]
        return im

    def dump(self, ix, iy, higlight = set()):
        for y in iy:
            for x in ix:
                if (x,y) in higlight:
                    print(end="\x1b[40;31m")
                print(end=".#"[self[x,y]])
                if (x,y) in higlight:
                    print(end="\x1b[m")
            print()
        print()


class Svg:
    def __init__(self, fname, width, height):
        self._f = open(fname, "w")
        self._print(
            f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="{width*8}" height="{height*8}">'
        )
        self._print(f'<rect width="{width*8}" height="{height*8}" style="fill:black"/>')

    def _print(self, *args, **kwargs):
        print(*args, **kwargs, file=self._f)

    def point(self, x, y):
        self._print(
            f'<rect x="{x*8}" y="{y*8}" width="7" height="7" style="fill:white"/>'
        )

    def annotation(self, x, y, w, h, text):
        self._print(
            f'<rect x="{x*8}" y="{y*8}" width="{w*8}" height="{h*8}" style="fill:green;opacity:0.5"/>'
        )
        style = "paint-order: stroke; fill: white; stroke: black; stroke-width: 2px; font:24px bold sans;"
        self._print(
            f'<text x="{x*8+w*4}" y="{y*8+h*4}" dominant-baseline="middle" text-anchor="middle" fill="white" style="{style}">{text}</text>'
        )

    def close(self):
        self._print("</svg>")
        self._f.close()


def decode_number(img, x, y):
    if img[x - 1, y - 1] or img[x, y - 1] or img[x - 1, y] or img[x, y]:
        return None

    # Get the size by iterating over top and left edges
    size = 0
    negative = False
    while True:
        items = (
            img[x + size + 1, y - 1],
            img[x + size + 1, y],
            img[x - 1, y + size + 1],
            img[x, y + size + 1],
        )
        if items == (False, True, False, True):
            size += 1
            continue
        if items == (False, False, False, False):
            break
        if items == (False, False, False, True):
            negative = True
            break
        return None

    if size == 0:
        return None

    # Check that right and bottom edges are empty
    for i in range(1,size + 2):
        if img[x + size + 1, y+i] or img[x+i, y + size + 1]:
            return None

    # Decode the number
    result, d = 0, 1
    for iy in range(size):
        for ix in range(size):
            result += d * img[x + ix + 1, y + iy + 1]
            d *= 2

    if negative:
        result = -result

    return (size, size+negative), result

M = 1
A = 0
N = -1

from copy import copy


syms = [
    ("=",
     [[M, M, M],
      [M, A, A],
      [M, M, M]
     ]),
    ("ap",
     [[M, M],
      [M, N]
    ])
]

def add_border(x):
    y = np.zeros((x.shape[0]+2, x.shape[1]+2))
    y.fill(N)
    y[1:-1, 1:-1] = x
    return y
syms  = [(name, add_border(np.array(f).transpose()))
         for (name, f) in syms]


filts = [
    [[N, M, N],
     [M, A, A],
     [N, A, A]
    ],
    [[N, M, M, N],
     [M, A, A, N],
     [M, A, A, N],
     [N, A, A, A],
    ],
    [[N, M, M, M, N],
     [M, A, A, A, N],
     [M, A, A, A, N],
     [M, A, A, A, N],
     [N, A, A, A, A],
    ],
    [[N, M, M, M, M, N],
     [M, A, A, A, A, N],
     [M, A, A, A, A, N],
     [M, A, A, A, A, N],
     [M, A, A, A, A, N],
     [N, A, A, A, A, A],
    ],
    [[N, M, M, M, M, M, N],
     [M, A, A, A, A, A, N],
     [M, A, A, A, A, A, N],
     [M, A, A, A, A, A, N],
     [M, A, A, A, A, A, N],
     [M, A, A, A, A, A, N],

     [N, A, A, A, A, A, A],
    ]


]
nfilts = []
for f in filts:
    f2 = [[y for y in x] for x in f]
    f2[-1][0] = 1
    nfilts.append(f2)

number_filters  = [np.array(f).transpose()
                   for f in filts]

neg_filters  = [np.array(f).transpose()
                for f in nfilts]

import math
def main(in_fname, out_fname):
    img = Img(in_fname, 4)
    svg = Svg(out_fname, img.size[0], img.size[1])

    n  = img.to_numpy()

    for y in range(img.size[1]):
        for x in range(img.size[0]):
            if img[x, y]:
                svg.point(x, y)

    def find(f):
        ori = (
            -math.floor(f.shape[0]/2),
            -math.floor(f.shape[1]/2)
        )
        out =  scipy.ndimage.filters.correlate(n, f,
                                               mode="constant",
                                               cval=0.0,
                                               origin = ori
        )
        return (out == (f == 1).sum()).nonzero()

    def add_sym(name, f):
        nz = find(f)
        for x, y in zip(nz[0].tolist(), nz[1].tolist()):
            svg.annotation(x, y , f.shape[0], f.shape[1], name)

    def add_num(off, f, neg=False):
        nz = find(f)
        for x, y in zip(nz[0].tolist(), nz[1].tolist()):
            vals = np.flip(n[x+1:x+off+1, y+1:y+off+1].transpose().reshape(-1))
            v = int(vals.dot(2**(np.arange(vals.size)[::-1])))
            svg.annotation(x, y, f.shape[0], f.shape[1], v if not neg else -v)

    for name, f in syms:
        add_sym(name, f)

    for off, f in enumerate(number_filters, 1):
        add_num(off, f, False)
    for off, f in enumerate(neg_filters, 1):
        add_num(off, f, True)


    svg.close()


main(sys.argv[1], sys.argv[2])
