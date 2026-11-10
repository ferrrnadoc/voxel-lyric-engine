"""kit.py - reusable blocky scenery builders for vox.World. All sizes in world units (1 = one block).
Facing for characters/props: one of '+z', '-z', '+x', '-x' (direction the FRONT looks)."""
import numpy as np
from vox import V, nrm, hexc, ascii_shape, text3d

FACE = {'+z': (V(0, 0, 1), V(1, 0, 0)), '-z': (V(0, 0, -1), V(-1, 0, 0)),
        '+x': (V(1, 0, 0), V(0, 0, -1)), '-x': (V(-1, 0, 0), V(0, 0, 1))}   # (front normal, local right)

def ground(W, x0, z0, x1, z1, y=0, cols=('#5fa83a', '#6bb544', '#589f35', '#73bd4b'), t_in=-1e9, flowers=0.0,
           flower_cols=('#ff6fa8', '#ffd84a', '#ffffff', '#9b7bff')):
    for x in range(int(x0), int(x1) + 1):
        for z in range(int(z0), int(z1) + 1):
            W.add((x, y, z), 1, cols[W.rng.integers(len(cols))], t_in=t_in, jitter=0.03)
            if flowers and W.rng.random() < flowers:
                W.add((x, y + 0.6, z), 0.22, flower_cols[W.rng.integers(len(flower_cols))], t_in=t_in, jitter=0)

def floor(W, x0, z0, x1, z1, y, pattern):
    """pattern(x, z) -> color or None."""
    for x in range(int(x0), int(x1) + 1):
        for z in range(int(z0), int(z1) + 1):
            c = pattern(x, z)
            if c: W.add((x, y, z), 1, c, jitter=0.04)

def tree(W, x, y, z, kind='oak', h=4, t_in=-1e9):
    trunk = {'oak': '#6b4a2b', 'cherry': '#5a3a2e', 'birch': '#e8e4d8'}[kind]
    leaf = {'oak': ('#3f8a2c', '#4b9a33', '#367a26'), 'cherry': ('#f6a9c8', '#f8bfd6', '#ef93b8'),
            'birch': ('#6aa84f', '#7bb85c', '#5a9940')}[kind]
    for i in range(h): W.add((x, y + 1 + i, z), 1, trunk, t_in=t_in)
    cy = y + h
    for dx in range(-2, 3):
        for dz in range(-2, 3):
            for dy in range(0, 3):
                r = abs(dx) + abs(dz) + dy * 1.3
                if r <= 3.4 and not (dx == 0 and dz == 0 and dy == 0):
                    W.add((x + dx, cy + dy, z + dz), 1, leaf[W.rng.integers(3)], t_in=t_in)

def clouds(W, n, x0, x1, z0, z1, y=28, s=2.0, col='#ffffff'):
    for _ in range(n):
        cx, cz = W.rng.uniform(x0, x1), W.rng.uniform(z0, z1)
        w, d = W.rng.integers(2, 6), W.rng.integers(2, 4)
        for i in range(w):
            for j in range(d):
                if W.rng.random() < 0.85: W.add((cx + i * s, y, cz + j * s), s, col, jitter=0.02)

def person(W, x, y, z, facing='+z', skin='#f2c39b', hair='#e0702c', shirt='#3dbb5c', pants='#3b3b8f',
           shoes='#4a4a4a', eye='#2a6fb5', px=0.0625, t_in=-1e9, anim='pop', long_hair=True, suit=False):
    """Minecraft-proportioned character (32 px tall, px world units per pixel). Feet at y."""
    N, R = FACE[facing]; base = V(x, y, z); piv = base + V(0, 16 * px, 0)
    def put(lx, ly, lz, c):   # local pixel coords: lx right, ly up, lz forward
        W.add(base + R * (lx + 0.5) * px + V(0, (ly + 0.5) * px, 0) + N * (lz + 0.5) * px, px, c, t_in=t_in, pivot=piv,
              anim=anim, jitter=0.04)
    def block(x0, y0, z0, sx, sy, sz, colf):
        for i in range(sx):
            for j in range(sy):
                for k in range(sz):
                    if 0 < i < sx - 1 and 0 < j < sy - 1 and 0 < k < sz - 1: continue
                    put(x0 + i, y0 + j, z0 + k, colf(i, j, k))
    jacket = '#2b2f4a' if suit else shirt
    block(-4, 0, -2, 4, 12, 4, lambda i, j, k: shoes if j < 2 else pants)        # right leg
    block(0, 0, -2, 4, 12, 4, lambda i, j, k: shoes if j < 2 else pants)         # left leg
    block(-4, 12, -2, 8, 12, 4, lambda i, j, k: ('#ffffff' if suit and k == 3 and 3 <= i <= 4 and j > 4 else
                                                 '#c0392b' if suit and k == 3 and 3 <= i <= 4 and j > 1 else jacket))
    block(-8, 12, -2, 4, 12, 4, lambda i, j, k: skin if j < 3 else jacket)        # arms
    block(4, 12, -2, 4, 12, 4, lambda i, j, k: skin if j < 3 else jacket)
    def head(i, j, k):
        if j >= 6 or (k < 2) or (long_hair and (i in (0, 7)) and j >= 1): return hair
        if k == 7:
            if j == 3 and i in (1, 6): return '#ffffff'
            if j == 3 and i in (2, 5): return eye
            if j == 1 and 3 <= i <= 4: return '#b5655a'
        return skin
    block(-4, 24, -4, 8, 8, 8, head)

def bench(W, x, y, z, length=4, facing='+z', col='#9c6b3c', leg='#4a4a4a'):
    N, R = FACE[facing]
    for i in range(length):
        p = V(x, y, z) + R * (i - (length - 1) / 2)
        W.add(p + V(0, 0.75, 0), 0.5, col); W.add(p + V(0, 0.75, 0) + R * 0.5 * 0 - N * 0.5, 0.5, col)
        W.add(p + V(0, 1.25, 0) - N * 0.75, 0.5, col); W.add(p + V(0, 1.75, 0) - N * 0.75, 0.5, col)
        if i in (0, length - 1): W.add(p + V(0, 0.25, 0), 0.5, leg)

def lamp(W, x, y, z, h=5, col='#2f2f36', light='#ffe9a8'):
    for i in range(h): W.add((x, y + 1 + i * 0.5, z), 0.5, col)
    W.add((x, y + 1 + h * 0.5, z), 0.75, light, emit=True)

def board(W, center, N, U, w, h, col='#a0703f', frame='#5e3f22', s=0.5, post=True, ground_y=None):
    """Flat sign / billboard. Returns the point just in front of the board centre (where to put text)."""
    N, U = nrm(N), nrm(U); R = np.cross(U, N); c = np.asarray(center, float)
    nx, ny = int(round(w / s)), int(round(h / s))
    for i in range(nx):
        for j in range(ny):
            edge = i in (0, nx - 1) or j in (0, ny - 1)
            W.add(c + R * (i - (nx - 1) / 2) * s + U * (j - (ny - 1) / 2) * s, s, frame if edge else col)
    if post and ground_y is not None:
        bottom = c - U * (h / 2)
        yy = bottom[1] - s
        while yy > ground_y:
            W.add((bottom[0], yy, bottom[2]), s, frame); yy -= s
    return c + N * (s / 2 + 0.05)

HEART = ['.XX...XX.', 'XXXX.XXXX', 'XXXXXXXXX', 'XXXXXXXXX', '.XXXXXXX.', '..XXXXX..', '...XXX...', '....X....']
BUBBLE = ['.XXXXXXXXXX.', 'XXXXXXXXXXXX', 'XXXXXXXXXXXX', 'XXXXXXXXXXXX', 'XXXXXXXXXXXX', '.XXXXXXXXXX.', '..XX........', '.X..........']
ARROW = ['....X...', '....XX..', 'XXXXXXX.', 'XXXXXXXX', 'XXXXXXX.', '....XX..', '....X...']
SPEAKER_MUTE = ['...X.......', '..XX..R...R', 'XXXX...R.R.', 'XXXX....R..', 'XXXX...R.R.', '..XX..R...R', '...X.......']
EYE = ['..WWWWW..', '.WWWWWWW.', 'WWWBBBWWW', 'WWWBKBWWW', 'WWWBBBWWW', '.WWWWWWW.', '..WWWWW..']
NOTEQ = ['......X', 'XXXXXX.', '....X..', 'XXXXXX.', '..X....', '.X.....']

def icon(W, rows, origin, N=(0, 0, 1), U=(0, 1, 0), vs=0.3, palette=None, depth=2, t_in=-1e9, anim='pop', **kw):
    palette = palette or {'X': '#ff4d6d'}
    ascii_shape(W, rows, origin, N, U, vs, palette, depth=depth, t_in=t_in, anim=anim, **kw)
