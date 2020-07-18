#!/usr/bin/env python
#! nix-shell -i python -p "python38.withPackages(p:[p.pillow])"

import sys
from PIL import Image
import numpy as np
import scipy.ndimage.filters
import math
from lang import *
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



M = 1
A = 0
N = -1

class Apply:
    def __init__(self, fn, val):
        self.fn = fn
        self.val = val

    def call(self):
        fn = self.fn
        while isinstance(fn, Apply):
            val = fn.val
            while isinstance(val, Apply):
                val = val.call(val)
                fn = fn()(val)
        return fn
    
from copy import copy
true_function  = lambda : lambda x: lambda : lambda y : x 
false_function = lambda : lambda x: lambda : lambda y : y 
cons_function = lambda : lambda x: lambda: lambda y : lambda: lambda z: (((z())(x))())(y)

def f38(protocol, ls):
    flag, newState, data = ls 
    if flag == 0:
        return (modem(newState), multipledraw(data))
    else:
        return interact(protocol, modem(newState), send(data))


    
syms = [
    ("=", None, 
     [[M, M, M],
      [M, A, A],
      [M, M, M]
     ]),
    ("ap", None, 
     [[M, M],
      [M, N]
     ]),
    ("add", lambda : lambda x : lambda :  lambda y : lambda: x() + y(),
     [[M, M, M, M],
      [M, M, N, M],
      [M, M, N, M],
      [M, M, N, M]
     ]),
    ("mul", lambda : lambda x : lambda : lambda y : lambda: x() * y(),
     [[M, M, M, M],
      [M, N, M, N],
      [M, N, M, N],
      [M, N, M, N]
     ]),
    ("lt", lambda :  lambda x : lambda :  lambda y :  true_function if x() < y() else false_function,
     [[M, M, M, M],
      [M, N, N, N],
      [M, N, N, M],
      [M, N, M, M]
     ]),
    ("div", lambda :  lambda x : lambda :  lambda y : lambda: x() // y(),
     [[M, M, M, M],
      [M, N, N, N],
      [M, M, N, M],
      [M, N, N, N]
     ]),
    ("eq", lambda : lambda x : lambda : lambda y : true_function if x() == y() else false_function,
     [[M, M, M, M],
      [M, N, N, N],
      [M, N, N, N],
      [M, M, M, M]
     ]),
    ("mod", None,
     [[M, M, M, M],
      [M, N, M, N],
      [M, M, N, M],
      [M, N, M, N]
     ]),
    ("dem", None,
     [[M, M, M, M],
      [M, M, N, M],
      [M, N, M, N],
      [M, M, N, M]
     ]),
    ("modem", None,
     [[M, M, M, M, M, M],
      [M, N, N, N, N, N],
      [M, N, N, N, N, N],
      [M, N, N, N, N, N],
      [M, M, N, N, M, M],
      [M, N, M, M, N, N],
     ]),
    ("coded", None,
     [[M, M, M, M, M, M],
      [M, M, M, M, M, M],
      [M, M, M, M, M, N],
      [M, M, M, N, N, N],
      [M, M, M, N, N, M],
      [M, M, N, N, N, N],
     ]),
    ("send", lambda : lambda x: lambda: x(),
     [[M, M, M, M],
      [M, N, M, M],
      [M, M, N, M],
      [M, N, M, N],
     ]),

    ("neg", lambda : lambda x : lambda: - x(),
     [[M, M, M],
      [M, N, M],
      [M, N, M]
     ]),
    # ("s", lambda : lambda x: lambda :  lambda y: lambda : lambda z: ((x()(z))())(y()(z)),
    #  [[M, M, M],
    #   [M, M, M],
    #   [M, M, N]
    #  ]),
    ("s", lambda : lambda x: lambda :  lambda y: lambda : lambda z: Apply(Apply(x, z), Apply(y, z)),
     [[M, M, M],
      [M, M, M],
      [M, M, N]
     ]),
    ("c", lambda : lambda x:lambda : lambda y: lambda : lambda z: ((x() (z))())(y),
     [[M, M, M],
      [M, N, M],
      [M, M, N]
     ]),
    ("b", lambda : lambda x: lambda : lambda y: lambda : lambda z: x() (y()(z)),
     [[M, M, M],
      [M, M, N],
      [M, M, N]
     ]),
    ("inc", lambda : lambda x: lambda: x() + 1, [[]]),
    ("dec", lambda : lambda x: lambda: x() - 1, [[]]),

    # ("kcomb",
    #  [[M, M, M],
    #   [M, N, M],
    #   [M, N, N]
    #  ]),
    ("i", lambda: lambda x: x,
     [[M, M],
      [M, M]
     ]),
    ("t", true_function,
     [[M, M, M],
      [M, N, M],
      [M, N, N]
     ]),
    ("f", false_function,
     [[M, M, M],
      [M, N, N],
      [M, N, M]
     ]),
    ("cons", lambda : lambda x: lambda: lambda y : lambda: lambda z: (((z())(x))())(y), 
     [[M, M, M, M, M],
      [M, N, M, N, M],
      [M, N, M, N, M],
      [M, N, M, N, M],      
      [M, M, M, M, M]
     ]),
    ("car", lambda : lambda x: x() (lambda : lambda a: lambda : lambda b: a),
     [[M, M, M, M, M],
      [M, N, M, M, M],
      [M, N, M, N, M],
      [M, N, M, N, M],      
      [M, M, M, M, M]
     ]),
    ("cdr", lambda : lambda x: x()(lambda : lambda a: lambda : lambda b: b),
     [[M, M, M, M, M],
      [M, M, M, N, M],
      [M, N, M, N, M],
      [M, N, M, N, M],      
      [M, M, M, M, M]
     ]),
    ("nil", lambda : lambda _: true_function,
     [[M, M, M],
      [M, N, M],
      [M, M, M]
     ]),
    ("isnil", lambda : lambda x: x()(lambda : lambda _: lambda: lambda _: false_function),
     [[M, M, M],
      [M, M, M],
      [M, M, M]
     ]),
    ("(", None,
     [[N, N, M],
      [N, M, M],
      [M, M, M],
      [N, M, M],
      [N, N, M]
     ]),

    (",", None, 
     [[M, M],
      [M, M],
      [M, M],
      [M, M],
      [M, M]
     ]),

    (")", None,
     [[M, N, N],
      [M, M, N],
      [M, M, M],
      [M, M, N],
      [M, N, N]
     ]),
    ("vec", cons_function,
     [[M, M, M, M, M, M],
      [M, M, N, N, N, N],
      [M, N, M, N, N, N],
      [M, N, N, M, N, N],      
      [M, N, N, N, M, N],
      [M, N, N, N, N, M]
     ]),
    ("draw", None,
     [[M, M, M, M, M, M],
      [M, N, N, N, N, M],
      [M, N, N, N, N, M],
      [M, N, N, N, N, M],      
      [M, N, N, N, N, M],
      [M, M, M, M, M, M]
     ]),
    ("checkerboard", None,
     [[M, M, M, M, M, M],
      [M, N, M, N, M, N],
      [M, M, N, M, N, M],
      [M, N, M, N, M, N],      
      [M, M, N, M, N, M],
      [M, N, M, N, M, N]
     ]),
    ("multipledraw", lambda : lambda x:  x,
     [[M, M, M, M, M, M, M],
      [M, N, N, M, N, N, M],
      [M, N, N, M, N, N, M],
      [M, M, M, M, M, M, M],
      [M, N, N, M, N, N, M],
      [M, N, N, M, N, N, M],
      [M, M, M, M, M, M, M],
     ]),

    ("pwr2", lambda : lambda x: lambda: 2 ** x(), 
     [[M, M, M, M, M, M, M],
      [M, N, N, N, N, N, M],
      [M, N, N, M, M, N, M],
      [M, N, M, N, M, N, M],
      [M, N, M, N, N, N, M],
      [M, N, N, N, N, N, M],
      [M, M, M, M, M, M, M],
     ]),
    ("if0", lambda : lambda x: true_function if 0 == x() else false_function, 
     [[M, M, M, M, M],
      [M, N, N, N, N],
      [M, N, M, M, M],
      [M, M, M, N, N],
      [M, N, M, M, M],
     ]),
    ("interact", lambda x: lambda y:  lambda z: f38(x, protocol(y, z)),
     [[M, M, M, M, M, M],
      [M, N, N, N, N, M],
      [M, N, M, M, N, M],
      [M, N, M, M, N, M],
      [M, N, N, N, N, M],
      [M, M, M, M, M, M],
     ]),
    ("statelessdraw", None,
     [[M, M, M, M, M, M, M],
      [M, N, N, N, N, M, N],
      [M, N, N, N, N, N, N],
      [M, N, N, N, N, N, N],
      [M, N, N, N, N, N, N],
      [M, N, M, N, N, N, N],
      [M, N, N, N, N, N, N],
     ]),
    ("statefuldraw", None,
     [[M, M, M, M, M, M, M],
      [M, M, N, N, N, N, N],
      [M, M, N, N, N, N, N],
      [M, N, N, N, N, N, N],
      [M, N, N, N, N, N, N],
      [M, N, N, M, N, N, N],
      [M, N, N, N, N, N, N],
     ]),

    ("\n...\n", None,
     [[M, N, M, N, M, N, M, N]
     ])

    # ("modlist",
    #  [[M, M, M, M, M],
    #   [M, N, M, N, M],
    #   [M, M, N, M, N],
    #   [M, N, M, N, M],      
    #   [M, M, N, M, N],

    #  ]),
]



def screen(sizea, size):
    shape = []
    top = [M for s in range(size)]
    top[0] = N
    top[-1] = N
    mid = [A for s in  range(size)]
    mid[0] = M
    mid[-1] = M
    shape.append(top)
    for _ in range(sizea-2):
        shape.append(mid)
    shape.append(top)
    return shape

for i in range(10, 40):
    syms.append(("|picture|", None, screen(i, 19)))

def add_border(x):
    y = np.zeros((x.shape[0]+2, x.shape[1]+2))
    y.fill(N)
    y[1:-1, 1:-1] = x
    return y
syms  = [(name, code, np.array(f).transpose())
         for (name, code, f) in syms]

vs = [
    [[M, M, M, M],
     [M, M, N, M],
     [M, N, A, M],
     [M, M, M, M],
    ],
    [[M, M, M, M, M],
     [M, M, N, N, M],
     [M, N, A, A, M],
     [M, N, A, A, M],
     [M, M, M, M, M],
    ],
    [[M, M, M, M, M, M],
     [M, M, N, N, N, M],
     [M, N, A, A, A, M],
     [M, N, A, A, A, M],
     [M, N, A, A, A, M],
     [M, M, M, M, M, M],
    ]
]

vs = [np.array(f).transpose()
        for f in vs]

incs = [
    [[M, M, M, M],
     [M, M, N, N],
     [M, N, A, A],
     [M, N, A, A]]
]

incs = [np.array(f).transpose()
        for f in incs]


filts = [
    [[N, M],
     [M, A],
     [N, A]
    ],
    [[N, M, M],
     [M, A, A],
     [M, A, A],
     [N, A, A],
    ],
    [[N, M, M, M],
     [M, A, A, A],
     [M, A, A, A],
     [M, A, A, A],
     [N, A, A, A],
    ],
    [[N, M, M, M, M],
     [M, A, A, A, A],
     [M, A, A, A, A],
     [M, A, A, A, A],
     [M, A, A, A, A],
     [N, A, A, A, A],
    ],
    [[N, M, M, M, M, M],
     [M, A, A, A, A, A],
     [M, A, A, A, A, A],
     [M, A, A, A, A, A],
     [M, A, A, A, A, A],
     [M, A, A, A, A, A],
     [N, A, A, A, A, A],
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

# binary_filters = []
# for i in range(4, 50):
#     out = np.zeros((i, 2))
#     out.fill(0)
#     for j, i in enumerate(r):
#         out[j, 1 if (i == "0") else 1] = M
#     binary_filters.append(out)


def search(n, f):
    ori = (
        -math.floor(f.shape[0]/2),
        -math.floor(f.shape[1]/2)
    )
    out =  scipy.ndimage.filters.correlate(n, f,
                                           mode="constant",
                                           cval=0.0,
                                           origin = ori
    )
    return out

def find(n, f):
    out = search(n, f)
    return (out == (f == 1).sum()).nonzero()

def convert_to_number(area):
    vals = np.flip(area.transpose().reshape(-1))
    v = int(vals.dot(2**(np.arange(vals.size)[::-1])))
    return v

def find_sym(n, name, f, extra=None):
    nz = find(n, f)
    items = []
    for x, y in zip(nz[0].tolist(), nz[1].tolist()):
        content = None
        if extra is not None:
            content = extra(n[x:x+f.shape[0], y:y+f.shape[1]])
            if content is None:
                continue
        sym = Sym(name, content)
        
        items.append(Item(x, y , f.shape[0], f.shape[1], sym))
    return items

def find_num(n, off, f, neg=False):
    nz = find(n ,f)
    items = []
    for x, y in zip(nz[0].tolist(), nz[1].tolist()):
        v = convert_to_number(n[x+1:x+off+1, y+1:y+off+1])
        content = Number(v if not neg else -v)
        items.append(Item(x, y, f.shape[0], f.shape[1], content))
    return items


def find_mod(n):
    out = (search(n, np.array([[-1, 1]])) == 1) + \
          (search(n, np.array([[1, -1]])) == 1)    
    
    items = []
    for i in range(3, 20):
        f = np.ones((i, 1))
        nz = find(out*1, f)
        for x, y in zip(nz[0].tolist(), nz[1].tolist()):
            vals = n[x:x+f.shape[0], y] 
            # print(vals)
            positive = (vals[1] == 1)
            width = 0
            for i in range(2, vals.shape[0]):
                width += 1
                if vals[i] == 0:
                    break

            vals = vals[i+1:]
            v = int(vals.dot(2**(np.arange(vals.size)[::-1]))) * (1 if positive else -1)
            items.append(Item(x, y, f.shape[0], 2, Sym("[", v)))
    return items


def main(in_fname, out_fname):
    img = Img(in_fname, 4)
    svg = Svg(out_fname, img.size[0], img.size[1])

    n  = img.to_numpy()

    for y in range(img.size[1]):
        for x in range(img.size[0]):
            if img[x, y]:
                svg.point(x, y)

    all_values = []

    for name, _, f in syms:
        all_values += find_sym(n, name, f)
        
    for off, f in enumerate(number_filters, 1):
        all_values += find_num(n, off, f, False)

    for off, f in enumerate(neg_filters, 1):
        all_values += find_num(n, off, f, True)


    def find_var_num(box):
        
        
        reverse_box = -1 * (box[2:-1, 2:-1] - 1)
        return convert_to_number(reverse_box)
    
    for f in vs:
        all_values += find_sym(n, "var", f, find_var_num)

    def find_inc_num(box):
        neg = (box[2, 2] == 1)
        return convert_to_number(box[3:, 3:] ) * (-1 if neg else 1)
    for f in incs:
        all_values += find_sym(n, "inc", f, find_inc_num)

    
    all_values += find_mod(n)

    seen = np.zeros(n.shape)
    seen[0, :] = 1
    seen[-1, :] = 1
    seen[:, 0] = 1
    seen[:, -1] = 1

    
    all_values.sort(key=lambda x: x.size[0] * x.size[1])
    all_values.reverse()
    kept = []
    for v in all_values:
        ran = seen[v.xy[0]: v.xy[0] + v.size[0], v.xy[1]: v.xy[1] + v.size[1]]
        if ran.sum() > 0:
            continue
        ran[:] = 1
        kept.append(v)
        v.render(svg)
        
    kept.sort(key=lambda x: (x.xy[1], x.xy[0]))
    last = None
    for v in kept:
        if last is not None and v.xy[1] > last.xy[1] + last.size[1]:
            print()
        print(v.content.write(), end=" ")
        last = v

        
    svg.close()

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
