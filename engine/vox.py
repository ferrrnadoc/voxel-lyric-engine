"""vox.py - tiny CPU voxel renderer for one-shot (no-cut) lyric videos in a blocky, Minecraft-like style.
numpy projection + PIL painter's-algorithm polygons. No GPU, no ML models.

Core ideas
- Everything (scenery AND lyrics) is voxels living in ONE world; the camera never cuts.
- Every voxel can have t_in / t_out, so lyric words "pop" into the world on the sung syllable.
- Camera = keyframes (t, pos, look, roll, fov) through Catmull-Rom; scenes are chained in space so
  the camera physically flies from one lyric line's scene into the next ("portal" transitions).
"""
import math, os, subprocess, numpy as np
from PIL import Image, ImageDraw, ImageFont

V = lambda *a: np.array(a, float)
UP = V(0, 1, 0)

def nrm(v):
    v = np.asarray(v, float); n = np.linalg.norm(v); return v / n if n > 1e-9 else v

def hexc(c):
    if isinstance(c, str):
        c = c.lstrip('#'); return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))
    return tuple(int(x) for x in c)

# ----------------------------------------------------------------------------- world
class World:
    """Voxel soup. add() appends cubes: center c, edge size s, color, timing, animation."""
    ANIMS = {'none': 0, 'pop': 1, 'drop': 2, 'rise': 3}

    def __init__(self, seed=7):
        self.c, self.s, self.col, self.tin, self.tout, self.piv, self.anim, self.emit, self.glow = [], [], [], [], [], [], [], [], []
        self.rng = np.random.default_rng(seed)

    def add(self, c, s=1.0, col='#ffffff', t_in=-1e9, t_out=1e9, pivot=None, anim='pop', emit=False, jitter=0.06, glow=False):
        """emit: flat colour (no face shading, e.g. neon/LED). glow: ignores ambient night darkening (lyrics)."""
        col = np.array(hexc(col), float)
        if jitter: col = np.clip(col * (1 + self.rng.uniform(-jitter, jitter)), 0, 255)
        self.c.append(np.asarray(c, float)); self.s.append(float(s)); self.col.append(col)
        self.tin.append(t_in); self.tout.append(t_out)
        self.piv.append(np.asarray(c if pivot is None else pivot, float)); self.anim.append(self.ANIMS[anim]); self.emit.append(bool(emit)); self.glow.append(bool(glow or emit))

    def box(self, p0, p1, col, s=1.0, hollow=True, **kw):
        """Fill integer-ish box [p0, p1] (inclusive, step s). hollow=True keeps only the shell."""
        xs = np.arange(p0[0], p1[0] + 1e-6, s); ys = np.arange(p0[1], p1[1] + 1e-6, s); zs = np.arange(p0[2], p1[2] + 1e-6, s)
        for a, x in enumerate(xs):
            for b, y in enumerate(ys):
                for c, z in enumerate(zs):
                    if hollow and 0 < a < len(xs) - 1 and 0 < b < len(ys) - 1 and 0 < c < len(zs) - 1:
                        continue
                    self.add((x, y, z), s, col if not callable(col) else col(x, y, z), **kw)

    def finalize(self):
        self.C = np.array(self.c); self.S = np.array(self.s); self.COL = np.array(self.col)
        self.TIN = np.array(self.tin); self.TOUT = np.array(self.tout); self.PIV = np.array(self.piv)
        self.ANIM = np.array(self.anim); self.EMIT = np.array(self.emit); self.GLOW = np.array(self.glow)
        n = len(self.C)
        # cull faces shared with an identical-timing, identical-size neighbour
        key = {}
        for i in range(n):
            k = (tuple(np.round(self.C[i] / self.S[i]).astype(int)), round(self.S[i], 4), self.TIN[i], self.TOUT[i])
            key[k] = i
        fv, fd = [], []
        for i in range(n):
            g = np.round(self.C[i] / self.S[i]).astype(int)
            for d, nv in enumerate(DIRS):
                k = (tuple(g + nv.astype(int)), round(self.S[i], 4), self.TIN[i], self.TOUT[i])
                if k in key: continue
                fv.append(i); fd.append(d)
        self.FV = np.array(fv, int); self.FD = np.array(fd, int)
        print(f'[vox] voxels={n} faces={len(self.FV)}')
        return self

DIRS = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
SHADE = np.array([0.80, 0.70, 1.00, 0.50, 0.90, 0.64])
def _corners():
    out = []
    for n in DIRS:
        a = np.roll(n, 1); b = np.cross(n, a)
        out.append([n + a + b, n + a - b, n - a - b, n - a + b])
    return np.array(out) * 0.5          # (6,4,3) unit-cube face corners
CORN = _corners()

# ----------------------------------------------------------------------------- shapes
def ascii_shape(W, rows, origin, N, U, vs, palette, depth=1, pivot=None, **kw):
    """Extrude ASCII art. rows: list of strings; palette: {'char': color}. ' ' or '.' = empty.
    N = direction the front faces (towards viewer), U = up on the art. Origin = art centre."""
    N, U = nrm(N), nrm(U); R = np.cross(U, N)
    h, w = len(rows), max(len(r) for r in rows); origin = np.asarray(origin, float)
    for i, r in enumerate(rows):
        for j, ch in enumerate(r):
            if ch in ' .' or ch not in palette: continue
            p = origin + R * (j - w / 2 + 0.5) * vs + U * (h / 2 - i - 0.5) * vs
            for k in range(depth):
                W.add(p - N * k * vs, vs, palette[ch], pivot=origin if pivot is None else pivot, **kw)

_FONTS = {}
def font(path, size):
    k = (path, size)
    if k not in _FONTS: _FONTS[k] = ImageFont.truetype(path, size)
    return _FONTS[k]

def _first(*paths):
    for p in paths:
        if os.path.exists(p): return p
    import glob
    hits = sorted(glob.glob('/usr/share/fonts/**/*Bold*.tt[fc]', recursive=True))
    return hits[0] if hits else paths[0]
BOLD = _first('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf')
CJK = _first('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc', '/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc')   # for Chinese/Japanese lyrics: fpath=CJK, px=14

def glyph_mask(text, fpath=BOLD, px=12):
    f = font(fpath, px); asc, desc = f.getmetrics()
    w = int(math.ceil(f.getlength(text))) + 2
    im = Image.new('L', (w, asc + desc + 2), 0); ImageDraw.Draw(im).text((1, 1), text, font=f, fill=255)
    m = np.array(im) > 100
    cols = np.where(m.any(0))[0]
    if len(cols) == 0: return m[:, :1], [0]
    m = m[:, cols[0]:cols[-1] + 1]
    bounds = [f.getlength(text[:k]) + 1 - cols[0] for k in range(len(text) + 1)]
    return m, bounds

def text3d(W, word, pos, N=(0, 0, 1), U=(0, 1, 0), height=1.5, col='#ffffff', t_in=-1e9, t_out=1e9,
           depth=2, outline='#1b1b2a', side=None, px=12, fpath=BOLD, per_letter=None, anim='pop',
           emit=False, build=0.0, tilt=0.0):
    """Voxel 3D word centred at pos. height = world height of the font box.
    per_letter=(t0, t1): letters appear one by one across [t0, t1] (for long/sustained words).
    build>0: voxels materialise individually over `build` seconds (block-by-block).
    tilt: rotate the word in its plane (degrees), handy for 'awkward'/'crooked' words.
    Returns the world-width of the word."""
    m, bounds = glyph_mask(word, fpath, px)
    N, U = nrm(N), nrm(U)
    if tilt:
        a = math.radians(tilt); R0 = np.cross(U, N); U = nrm(U * math.cos(a) + R0 * math.sin(a))
    R = np.cross(U, N)
    h, w = m.shape; vs = height / h; pos = np.asarray(pos, float)
    if outline is not None:
        from scipy.ndimage import binary_dilation
        ring = binary_dilation(m, iterations=1) & ~m
    side = side or tuple(int(x * 0.55) for x in hexc(col))
    for i in range(h):
        for j in range(w):
            on = m[i, j]; ol = outline is not None and ring[i, j]
            if not (on or ol): continue
            p = pos + R * (j - w / 2 + 0.5) * vs + U * (h / 2 - i - 0.5) * vs
            ti = t_in
            if per_letter:
                k = max(0, np.searchsorted(bounds, j + 0.5) - 1)
                ti = per_letter[0] + (per_letter[1] - per_letter[0]) * k / max(1, len(word) - 1)
            if build: ti = ti + W.rng.uniform(0, build)
            piv = p if build else pos
            if on:
                W.add(p, vs, col, ti, t_out, piv, anim, emit, jitter=0.03, glow=True)
                for k in range(1, depth): W.add(p - N * k * vs, vs, side, ti, t_out, piv, anim, emit, jitter=0.02, glow=True)
            else:
                for k in range(0, depth): W.add(p - N * (k + 0.5) * vs, vs, outline, ti, t_out, piv, anim, False, jitter=0.0, glow=True)
    return w * vs

def word_row(W, words, pos, N=(0, 0, 1), U=(0, 1, 0), height=1.2, gap=0.35, max_width=None, line_gap=0.25, **kw):
    """Lay words left->right (wrapping) centred on pos. words: list of dicts {w, t, (col, height, ...)}.
    Each word appears at its own t (the sung time)."""
    N, U = nrm(N), nrm(U); R = np.cross(U, N)
    widths = []
    for d in words:
        m, _ = glyph_mask(d['w'], kw.get('fpath', BOLD), kw.get('px', 12)); widths.append(m.shape[1] * d.get('height', height) / m.shape[0])
    rows, cur, cw = [], [], 0
    for d, wd in zip(words, widths):
        if max_width and cur and cw + gap + wd > max_width: rows.append(cur); cur, cw = [], 0
        cur.append((d, wd)); cw += wd + (gap if len(cur) > 1 else 0)
    if cur: rows.append(cur)
    y = (len(rows) - 1) / 2 * (height + line_gap)
    for row in rows:
        tw = sum(wd for _, wd in row) + gap * (len(row) - 1); x = -tw / 2
        for d, wd in row:
            p = np.asarray(pos, float) + R * (x + wd / 2) + U * y
            opts = dict(kw); opts.update({k: v for k, v in d.items() if k not in ('w', 't')})
            opts.setdefault('height', height)
            text3d(W, d['w'], p, N, U, t_in=d['t'], **opts); x += wd + gap
        y -= height + line_gap

# ----------------------------------------------------------------------------- camera
class Camera:
    """keys: list of dict(t, pos, look, roll=0, fov=60). Catmull-Rom through pos/look, linear roll/fov.
    bumps: optional [(t, strength), ...] beat punch-ins (fov shrinks by strength, decays in ~0.1 s), e.g.
    [(b, 0.035) for b in WJ['beats'] if chorus_t0 <= b < chorus_t1] for drum-driven choruses."""
    def __init__(self, keys, bumps=None):
        self.k = sorted(keys, key=lambda d: d['t']); self.b = sorted(bumps or [])
    def _cr(self, name, t):
        ks = self.k; ts = [d['t'] for d in ks]
        if t <= ts[0]: return np.asarray(ks[0][name], float)
        if t >= ts[-1]: return np.asarray(ks[-1][name], float)
        i = np.searchsorted(ts, t) - 1
        p = [np.asarray(ks[max(0, min(len(ks) - 1, j))][name], float) for j in (i - 1, i, i + 1, i + 2)]
        u = (t - ts[i]) / (ts[i + 1] - ts[i]); u = u * u * (3 - 2 * u) * 0.35 + u * 0.65   # gentle ease
        return 0.5 * ((2 * p[1]) + (-p[0] + p[2]) * u + (2 * p[0] - 5 * p[1] + 4 * p[2] - p[3]) * u * u
                      + (-p[0] + 3 * p[1] - 3 * p[2] + p[3]) * u ** 3)
    def _lin(self, name, t, default):
        ks = self.k; ts = [d['t'] for d in ks]; vs = [d.get(name, default) for d in ks]
        return float(np.interp(t, ts, vs))
    def at(self, t):
        fov = self._lin('fov', t, 60.0)
        for tb, st in self.b:
            d = t - tb
            if 0 <= d < 0.4: fov *= 1 - st * math.exp(-d / 0.09)
        return self._cr('pos', t), self._cr('look', t), self._lin('roll', t, 0.0), fov

# ----------------------------------------------------------------------------- sky / grading
class Sky:
    """keys: list of (t, top_color, bottom_color[, stars 0..1[, ambient light 0..1]])."""
    def __init__(self, keys): self.k = sorted(keys, key=lambda x: x[0])
    def at(self, t):
        ts = [k[0] for k in self.k]
        f = lambda idx: np.array([np.interp(t, ts, [hexc(k[idx])[c] for k in self.k]) for c in range(3)])
        stars = float(np.interp(t, ts, [k[3] if len(k) > 3 else 0 for k in self.k]))
        light = float(np.interp(t, ts, [k[4] if len(k) > 4 else 1.0 for k in self.k]))
        return f(1), f(2), stars, light

def _ease_back(x):
    c1 = 1.70158; c3 = c1 + 1; x = np.clip(x, 0, 1)
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2

# ----------------------------------------------------------------------------- renderer
G = {}
def _render_frame(fi):
    W, cam, sky, cfg = G['W'], G['cam'], G['sky'], G['cfg']
    t = cfg['t0'] + fi / cfg['fps']; S = cfg['size']
    pos, look, roll, fov = cam.at(t)
    f = nrm(look - pos); up = UP if abs(f[1]) < 0.995 else V(0, 0, 1)
    r = nrm(np.cross(f, up)); u = np.cross(r, f)
    a = math.radians(roll); r, u = r * math.cos(a) + u * math.sin(a), u * math.cos(a) - r * math.sin(a)
    focal = (S / 2) / math.tan(math.radians(fov) / 2)
    top, bot, stars, light = sky.at(t)
    # background gradient
    g = np.linspace(0, 1, S)[:, None, None]
    bg = (top[None, None, :] * (1 - g) + bot[None, None, :] * g) * np.ones((1, S, 1))
    img = Image.fromarray(bg.astype(np.uint8))
    dr = ImageDraw.Draw(img)
    if stars > 0.01:
        rs = np.random.default_rng(3); sp = rs.uniform(-1, 1, (900, 3)); sp = sp / np.linalg.norm(sp, axis=1)[:, None]
        z = sp @ f; ok = z > 0.2
        x = S / 2 + focal * (sp[ok] @ r) / z[ok]; y = S / 2 - focal * (sp[ok] @ u) / z[ok]
        for xx, yy in zip(x, y):
            if 0 <= xx < S and 0 <= yy < S: dr.rectangle([xx, yy, xx + 1, yy + 1], fill=tuple(int(255 * stars) for _ in range(3)))
    # animated voxel state
    ti, to, A = W.TIN, W.TOUT, W.ANIM
    dur = cfg['pop']
    kin = np.where(A == 0, (t >= ti).astype(float), _ease_back((t - ti) / dur))
    kout = np.clip(1 - (t - to) / dur, 0, 1)
    k = np.minimum(kin, kout); k = np.where(t < ti, 0.0, k)
    C = W.PIV + (W.C - W.PIV) * k[:, None]
    lift = np.where(A == 2, (1 - np.clip((t - ti) / 0.35, 0, 1)) ** 2 * 3.0, 0) - np.where(A == 3, (1 - np.clip((t - ti) / 0.35, 0, 1)) ** 2 * 3.0, 0)
    C = C + np.outer(lift, UP)
    Sz = W.S * k
    fv, fd = W.FV, W.FD
    alive = Sz[fv] > 1e-3
    fv, fd = fv[alive], fd[alive]
    fc = C[fv] + DIRS[fd] * (Sz[fv] / 2)[:, None]
    toc = pos[None, :] - fc
    dist = np.linalg.norm(toc, axis=1)
    keep = (np.einsum('ij,ij->i', DIRS[fd], toc) > 0) & (dist < cfg['far'])
    fv, fd, fc, dist = fv[keep], fd[keep], fc[keep], dist[keep]
    P = C[fv][:, None, :] + CORN[fd] * Sz[fv][:, None, None]          # (n,4,3)
    D = P - pos
    cx, cy, cz = D @ r, D @ u, D @ f
    ok = (cz > cfg['near']).all(1)
    sx = S / 2 + focal * cx / np.maximum(cz, 1e-6); sy = S / 2 - focal * cy / np.maximum(cz, 1e-6)
    ok &= ~((sx < 0).all(1) | (sx > S).all(1) | (sy < 0).all(1) | (sy > S).all(1))
    ok &= (np.abs(sx) < 50 * S).all(1) & (np.abs(sy) < 50 * S).all(1)
    fv, fd, dist, sx, sy = fv[ok], fd[ok], dist[ok], sx[ok], sy[ok]
    col = W.COL[fv] * (np.where(W.EMIT[fv], 1.0, SHADE[fd]) * np.where(W.GLOW[fv], 1.0, light))[:, None]
    fog = np.clip((dist - cfg['fog'][0]) / (cfg['fog'][1] - cfg['fog'][0]), 0, 1)[:, None]
    fog = np.where(W.EMIT[fv][:, None], fog * 0.4, fog)
    col = np.clip(col * (1 - fog) + bot[None, :] * fog, 0, 255).astype(int)
    order = np.argsort(-dist)
    ol = np.where((W.S[fv] >= 0.5)[:, None], col * 0.86, col).astype(int)   # faint block grid on scenery
    xy = np.stack([sx, sy], -1)[order].round(1).tolist(); cl = col[order].tolist(); oc = ol[order].tolist()
    for q, c, o in zip(xy, cl, oc):
        dr.polygon([tuple(p) for p in q], fill=tuple(c), outline=tuple(o))
    arr = np.asarray(img).astype(np.float32)
    arr *= G['vig']
    fade = 1.0
    if cfg['fade_in'] > 0: fade = min(fade, (t - cfg['t0']) / cfg['fade_in'])
    if cfg['fade_out'] > 0: fade = min(fade, (cfg['t1'] - t) / cfg['fade_out'])
    arr *= np.clip(fade, 0, 1)
    return np.clip(arr, 0, 255).astype(np.uint8).tobytes()

def render(W, cam, sky, out, audio=None, t0=0.0, t1=10.0, fps=30, size=1080, workers=None, near=0.15,
           far=90, fog=(35, 85), pop=0.16, fade_in=0.3, fade_out=0.5, audio_offset=0.0, crf=18, frames=None):
    """Render [t0, t1) to mp4 (H.264, AAC). audio_offset = where t0 sits inside the audio file."""
    yy, xx = np.mgrid[0:size, 0:size] / size - 0.5
    vig = (1 - 0.35 * (xx ** 2 + yy ** 2) ** 1.2 * 2.2)[:, :, None].astype(np.float32)
    G.update(W=W, cam=cam, sky=sky, vig=vig, cfg=dict(t0=t0, t1=t1, fps=fps, size=size, near=near, far=far, fog=fog,
                                                         pop=pop, fade_in=fade_in, fade_out=fade_out))
    idx = list(range(int(round((t1 - t0) * fps)))) if frames is None else frames
    if frames is not None:   # preview stills
        for fi in frames:
            Image.frombytes('RGB', (size, size), _render_frame(fi)).save(f'{out}_{fi:05d}.png')
        return
    cmd = ['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{size}x{size}', '-r', str(fps), '-i', '-']
    if audio: cmd += ['-ss', str(audio_offset + t0), '-t', str(t1 - t0), '-i', audio, '-map', '0:v', '-map', '1:a',
                      '-af', f'afade=t=out:st={max(0, t1 - t0 - fade_out)}:d={fade_out}', '-c:a', 'aac', '-b:a', '192k']
    cmd += ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', str(crf), '-pix_fmt', 'yuv420p', '-shortest', out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    import multiprocessing as mp
    workers = workers or os.cpu_count()
    with mp.get_context('fork').Pool(workers) as pool:
        for n, fr in enumerate(pool.imap(_render_frame, idx, chunksize=4)):
            p.stdin.write(fr)
            if n % 60 == 0: print(f'[vox] frame {n}/{len(idx)}', flush=True)
    p.stdin.close(); p.wait()
    print('[vox] wrote', out)
