"""vox2.py - extensions to vox.py for richer lyric staging.
New:
- World.add(..., off=(dx,dy,dz), md=sec, arch=h): anim='move' flies a voxel in from c+off to c over md seconds
  (ease-out, optional parabolic arc of height arch). Good for letters rising, leaves gathering, scatter->focus.
- World.group(fn): rigid groups. fn(t) -> (M 3x3, pivot, T) applied to every voxel added with grp=id
  (rotate/scale about pivot, then translate). Good for page turns, spinning tags, moving cars, clock hands.
- World.add(..., bias=b): painter's-sort bias; use 0.5-0.8 for voxels lying flat on big blocks (road paint, decals).
- Shading from the (possibly rotated) face normal; sky sun disc; fonts with variable weight and threshold.
"""
import math, os, subprocess, numpy as np
from PIL import Image, ImageDraw, ImageFont
from vox import V, UP, nrm, hexc, DIRS, CORN, Camera, Sky, _ease_back

S_POS = np.array([0.80, 1.00, 0.90]); S_NEG = np.array([0.70, 0.50, 0.64])


def shade_of(n):
    a = np.abs(n) + 1e-9
    return (np.where(n > 0, S_POS, S_NEG) * a).sum(-1) / a.sum(-1)


def rot(axis, deg):
    ax = nrm(axis); a = math.radians(deg); c, s = math.cos(a), math.sin(a); x, y, z = ax
    return np.array([[c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                     [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
                     [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)]])


def ease(u): u = np.clip(u, 0, 1); return 1 - (1 - u) ** 3
def smooth(u): u = min(max(u, 0.0), 1.0); return u * u * (3 - 2 * u)


class World:
    ANIMS = {'none': 0, 'pop': 1, 'drop': 2, 'rise': 3, 'move': 4}

    def __init__(self, seed=7):
        self.L = {k: [] for k in ('c', 's', 'col', 'tin', 'tout', 'piv', 'anim', 'emit', 'glow', 'off', 'md', 'arch', 'grp', 'bias', 'mid', 'gk')}
        self.rng = np.random.default_rng(seed); self.groups = []
        self.mats = [np.eye(3)]; self._matidx = {}
        self.words = []   # registry for readcheck: dict(w, t_full, t_out, pos, height, N, U, tier, line)

    def mat(self, M):
        """Index of a static voxel orientation (3x3, columns = local x,y,z in world). 0 = axis-aligned."""
        if M is None: return 0
        M = np.asarray(M, float); k = tuple(np.round(M, 5).ravel())
        if k not in self._matidx: self.mats.append(M); self._matidx[k] = len(self.mats) - 1
        return self._matidx[k]

    def group(self, fn):
        self.groups.append(fn); return len(self.groups) - 1

    def add(self, c, s=1.0, col='#ffffff', t_in=-1e9, t_out=1e9, pivot=None, anim='pop', emit=False, jitter=0.06,
            glow=False, off=None, md=0.6, arch=0.0, grp=-1, bias=0.0, M=None, gk=None):
        """M: 3x3 orientation of this cube (columns = its local x,y,z); gk: integer grid key in that local frame
        (e.g. (letter_id, col, row, layer)) so hidden faces between rotated cubes can still be culled."""
        col = np.array(hexc(col), float)
        if jitter: col = np.clip(col * (1 + self.rng.uniform(-jitter, jitter)), 0, 255)
        L = self.L
        L['c'].append(np.asarray(c, float)); L['s'].append(float(s)); L['col'].append(col)
        L['tin'].append(t_in); L['tout'].append(t_out)
        L['piv'].append(np.asarray(c if pivot is None else pivot, float)); L['anim'].append(self.ANIMS[anim])
        L['emit'].append(bool(emit)); L['glow'].append(bool(glow or emit))
        L['off'].append(np.zeros(3) if off is None else np.asarray(off, float)); L['md'].append(float(md))
        L['arch'].append(float(arch)); L['grp'].append(int(grp)); L['bias'].append(float(bias))
        L['mid'].append(self.mat(M) if M is not None and not isinstance(M, (int, np.integer)) else int(M or 0)); L['gk'].append(gk)

    def box(self, p0, p1, col, s=1.0, hollow=True, **kw):
        xs = np.arange(p0[0], p1[0] + 1e-6, s); ys = np.arange(p0[1], p1[1] + 1e-6, s); zs = np.arange(p0[2], p1[2] + 1e-6, s)
        for a, x in enumerate(xs):
            for b, y in enumerate(ys):
                for c, z in enumerate(zs):
                    if hollow and 0 < a < len(xs) - 1 and 0 < b < len(ys) - 1 and 0 < c < len(zs) - 1: continue
                    self.add((x, y, z), s, col if not callable(col) else col(x, y, z), **kw)

    def finalize(self):
        L = self.L
        self.C = np.array(L['c']); self.S = np.array(L['s']); self.COL = np.array(L['col'])
        self.TIN = np.array(L['tin']); self.TOUT = np.array(L['tout']); self.PIV = np.array(L['piv'])
        self.ANIM = np.array(L['anim']); self.EMIT = np.array(L['emit']); self.GLOW = np.array(L['glow'])
        self.OFF = np.array(L['off']); self.MD = np.array(L['md']); self.ARCH = np.array(L['arch']); self.GRP = np.array(L['grp'])
        self.BIAS = np.array(L['bias'])   # painter's-sort bias: decals lying on big blocks draw after them
        self.MID = np.array(L['mid'], int); self.MATS = np.array(self.mats)
        n = len(self.C); key = {}; gkey = {}
        mov = self.ANIM == 4
        for i in range(n):
            if mov[i]: continue
            if L['gk'][i] is not None:
                gkey[(tuple(L['gk'][i]), self.MID[i], self.TIN[i], self.TOUT[i], self.GRP[i])] = i; continue
            if self.MID[i]: continue
            k = (tuple(np.round(self.C[i] / self.S[i]).astype(int)), round(self.S[i], 4), self.TIN[i], self.TOUT[i], self.GRP[i])
            key[k] = i
        fv, fd = [], []
        for i in range(n):
            gk = L['gk'][i]
            g = np.round(self.C[i] / self.S[i]).astype(int)
            for d, nv in enumerate(DIRS):
                if not mov[i]:
                    if gk is not None:
                        nk = tuple(gk[:-3]) + tuple(np.asarray(gk[-3:]) + nv.astype(int))
                        if (nk, self.MID[i], self.TIN[i], self.TOUT[i], self.GRP[i]) in gkey: continue
                    elif not self.MID[i]:
                        k = (tuple(g + nv.astype(int)), round(self.S[i], 4), self.TIN[i], self.TOUT[i], self.GRP[i])
                        if k in key: continue
                fv.append(i); fd.append(d)
        self.FV = np.array(fv, int); self.FD = np.array(fd, int)
        print(f'[vox2] voxels={n} faces={len(self.FV)} groups={len(self.groups)}')
        return self


# ----------------------------------------------------------------------------- fonts / text
_FONTS = {}
FONTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fonts')


def font(name, px, wght=None):
    k = (name, px, wght)
    if k not in _FONTS:
        path = name if os.path.isabs(name) else os.path.join(FONTDIR, name)
        f = ImageFont.truetype(path, px)
        if wght is not None and '[' in os.path.basename(path):      # static fonts ignore wght
            axes = f.get_variation_axes(); vals = [a['default'] for a in axes]
            for i, a in enumerate(axes):
                if a['name'] in (b'Weight', 'Weight'): vals[i] = wght
            f.set_variation_by_axes(vals)
        _FONTS[k] = f
    return _FONTS[k]


def glyph_mask(text, name, px=12, wght=None, thr=100):
    f = font(name, px, wght); asc, desc = f.getmetrics()
    w = int(math.ceil(f.getlength(text))) + 2
    im = Image.new('L', (w, asc + desc + 2), 0); ImageDraw.Draw(im).text((1, 1), text, font=f, fill=255)
    m = np.array(im) > thr
    cols = np.where(m.any(0))[0]
    if len(cols) == 0: return m[:1, :1], [0]
    m = m[:, cols[0]:cols[-1] + 1]
    bounds = [f.getlength(text[:k]) + 1 - cols[0] for k in range(len(text) + 1)]
    return m, bounds


def text3d(W, word, pos, N=(0, 0, 1), U=(0, 1, 0), height=1.5, col='#ffffff', t_in=-1e9, t_out=1e9, depth=2,
           outline='#1b1b2a', side=None, font_name='Fredoka[wdth,wght].ttf', px=14, wght=None, thr=100,
           per_letter=None, anim='pop', emit=False, build=0.0, tilt=0.0, grp=-1, off=None, md=0.6, arch=0.0,
           letter_off=None, glow=True, letter_t=None, letter_tout=None, bias=0.0):
    """Voxel word centred at pos. letter_off(k) -> (off, t_in_delay) lets each letter fly in on its own path.
    letter_t / letter_tout: explicit per-character appear / disappear times (typing and un-typing, see layout.py)."""
    m, bounds = glyph_mask(word, font_name, px, wght, thr)
    N, U = nrm(N), nrm(U)
    if tilt:
        a = math.radians(tilt); R0 = np.cross(U, N); U = nrm(U * math.cos(a) + R0 * math.sin(a))
    R = np.cross(U, N)
    h, w = m.shape; vs = height / h; pos = np.asarray(pos, float)
    ring = None
    if outline is not None:
        from scipy.ndimage import binary_dilation
        ring = binary_dilation(m, iterations=1) & ~m
    side = side or tuple(int(x * 0.55) for x in hexc(col))
    for i in range(h):
        for j in range(w):
            on = m[i, j]; ol = ring is not None and ring[i, j]
            if not (on or ol): continue
            p = pos + R * (j - w / 2 + 0.5) * vs + U * (h / 2 - i - 0.5) * vs
            k = max(0, int(np.searchsorted(bounds, j + 0.5)) - 1)
            ti = t_in; o = off
            if per_letter: ti = per_letter[0] + (per_letter[1] - per_letter[0]) * k / max(1, len(word) - 1)
            if letter_t is not None: ti = letter_t[min(k, len(letter_t) - 1)]
            to_k = letter_tout[min(k, len(letter_tout) - 1)] if letter_tout is not None else t_out
            if letter_off is not None:
                lo, dl = letter_off(k); o = lo; ti = ti + dl
            if build: ti = ti + W.rng.uniform(0, build)
            piv = p if (build or anim == 'move') else pos
            kw = dict(grp=grp, off=o, md=md, arch=arch, bias=bias)
            if on:
                W.add(p, vs, col, ti, to_k, piv, anim, emit, jitter=0.03, glow=glow, **kw)
                for d in range(1, depth): W.add(p - N * d * vs, vs, side, ti, to_k, piv, anim, emit, jitter=0.02, glow=glow, **kw)
            else:
                for d in range(depth): W.add(p - N * (d + 0.5) * vs, vs, outline, ti, to_k, piv, anim, False, jitter=0.0, glow=glow, **kw)
    return w * vs


def mask_points(word, pos, N=(0, 0, 1), U=(0, 1, 0), height=1.5, font_name='Fredoka[wdth,wght].ttf', px=14, wght=None, thr=100):
    """'On' pixels of a word in world space, to build custom-material letters.
    Returns (points, voxel_size, (w, h)); points = [(pos, letter_idx, col, row), ...]; height = full line box."""
    m, bounds = glyph_mask(word, font_name, px, wght, thr)
    N, U = nrm(N), nrm(U); R = np.cross(U, N); h, w = m.shape; vs = height / h; pos = np.asarray(pos, float)
    out = []
    for i in range(h):
        for j in range(w):
            if m[i, j]:
                k = max(0, int(np.searchsorted(bounds, j + 0.5)) - 1)
                out.append((pos + R * (j - w / 2 + 0.5) * vs + U * (h / 2 - i - 0.5) * vs, k, j, i))
    return out, vs, (w, h)


def ascii_shape(W, rows, origin, N, U, vs, palette, depth=1, pivot=None, **kw):
    N, U = nrm(N), nrm(U); R = np.cross(U, N)
    h, w = len(rows), max(len(r) for r in rows); origin = np.asarray(origin, float)
    for i, r in enumerate(rows):
        for j, ch in enumerate(r):
            if ch in ' .' or ch not in palette: continue
            p = origin + R * (j - w / 2 + 0.5) * vs + U * (h / 2 - i - 0.5) * vs
            for k in range(depth):
                W.add(p - N * k * vs, vs, palette[ch], pivot=origin if pivot is None else pivot, **kw)


# ----------------------------------------------------------------------------- renderer
G = {}


def _render_core(t):
    W, cam, sky, cfg = G['W'], G['cam'], G['sky'], G['cfg']
    S = cfg['size']
    pos, look, roll, fov = cam.at(t)
    f = nrm(look - pos); up = UP if abs(f[1]) < 0.995 else V(0, 0, 1)
    r = nrm(np.cross(f, up)); u = np.cross(r, f)
    a = math.radians(roll); r, u = r * math.cos(a) + u * math.sin(a), u * math.cos(a) - r * math.sin(a)
    focal = (S / 2) / math.tan(math.radians(fov) / 2)
    top, bot, stars, light = sky.at(t)
    g = np.linspace(0, 1, S)[:, None, None]
    bg = (top[None, None, :] * (1 - g) + bot[None, None, :] * g) * np.ones((1, S, 1))
    img = Image.fromarray(bg.astype(np.uint8)); dr = ImageDraw.Draw(img)
    if stars > 0.01:
        rs = np.random.default_rng(3); sp = rs.uniform(-1, 1, (900, 3)); sp = sp / np.linalg.norm(sp, axis=1)[:, None]
        z = sp @ f; ok = (z > 0.2) & (sp[:, 1] > 0.02)
        x = S / 2 + focal * (sp[ok] @ r) / z[ok]; y = S / 2 - focal * (sp[ok] @ u) / z[ok]
        for xx, yy in zip(x, y):
            if 0 <= xx < S and 0 <= yy < S: dr.rectangle([xx, yy, xx + 1, yy + 1], fill=tuple(int(255 * stars) for _ in range(3)))
    sun = cfg.get('sun')
    if sun:
        sd = sun(t)
        if sd:
            d, ang, col, halo = sd; d = nrm(d); z = d @ f
            if z > 0.05:
                cx = S / 2 + focal * (d @ r) / z; cy = S / 2 - focal * (d @ u) / z; rad = focal * math.tan(math.radians(ang))
                hc = tuple(int(0.5 * hexc(col)[k] + 0.5 * bot[k]) for k in range(3))
                dr.ellipse([cx - rad * halo, cy - rad * halo, cx + rad * halo, cy + rad * halo], fill=hc)
                dr.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=hexc(col))
    ti, to, A = W.TIN, W.TOUT, W.ANIM
    dur = cfg['pop']
    kin = np.where((A == 0) | (A == 4), (t >= ti).astype(float), _ease_back((t - ti) / dur))
    kout = np.clip(1 - (t - to) / dur, 0, 1)
    k = np.minimum(kin, kout); k = np.where(t < ti, 0.0, k)
    C = W.PIV + (W.C - W.PIV) * k[:, None]
    lt = np.clip((t - ti) / 0.35, 0, 1)
    lift = np.where(A == 2, (1 - lt) ** 2 * 3.0, 0) - np.where(A == 3, (1 - lt) ** 2 * 3.0, 0)
    C = C + np.outer(lift, UP)
    mv = A == 4
    if mv.any():
        uu = np.clip((t - ti[mv]) / np.maximum(W.MD[mv], 1e-3), 0, 1)
        e = ease(uu)
        C[mv] = W.C[mv] + W.OFF[mv] * (1 - e)[:, None] + np.outer(W.ARCH[mv] * 4 * uu * (1 - uu), UP)
    Sz = W.S * np.where(mv, (t >= ti) & (t < to + dur), 1.0) * np.where(mv, 1.0, k)
    Sz = np.where(mv & (t >= to), W.S * kout, Sz)
    fv, fd = W.FV, W.FD
    alive = Sz[fv] > 1e-3
    fv, fd = fv[alive], fd[alive]
    nrmv = DIRS[fd].copy(); off = CORN[fd].copy()
    rm = W.MID[fv] > 0
    if rm.any():
        Mf = W.MATS[W.MID[fv[rm]]]
        off[rm] = np.einsum('nij,nkj->nki', Mf, off[rm]); nrmv[rm] = np.einsum('nij,nj->ni', Mf, nrmv[rm])
    P = C[fv][:, None, :] + off * Sz[fv][:, None, None]
    gr = W.GRP[fv]
    for gi, fn in enumerate(W.groups):
        gm = gr == gi
        if not gm.any(): continue
        M, pv, T = fn(t)
        M = np.asarray(M, float); pv = np.asarray(pv, float); T = np.asarray(T, float)
        P[gm] = (P[gm] - pv) @ M.T + pv + T
        nn = nrmv[gm] @ M.T; nrmv[gm] = nn / (np.linalg.norm(nn, axis=1)[:, None] + 1e-9)
    fc = P.mean(1)
    toc = pos[None, :] - fc
    dist = np.linalg.norm(toc, axis=1)
    keep = (np.einsum('ij,ij->i', nrmv, toc) > 0) & (dist < cfg['far'])
    fv, fd, P, nrmv, dist = fv[keep], fd[keep], P[keep], nrmv[keep], dist[keep]
    D = P - pos
    cx, cy, cz = D @ r, D @ u, D @ f
    ok = (cz > cfg['near']).all(1)
    sx = S / 2 + focal * cx / np.maximum(cz, 1e-6); sy = S / 2 - focal * cy / np.maximum(cz, 1e-6)
    ok &= ~((sx < 0).all(1) | (sx > S).all(1) | (sy < 0).all(1) | (sy > S).all(1))
    ok &= (np.abs(sx) < 50 * S).all(1) & (np.abs(sy) < 50 * S).all(1)
    fv, nrmv, dist, sx, sy = fv[ok], nrmv[ok], dist[ok], sx[ok], sy[ok]
    col = W.COL[fv] * (np.where(W.EMIT[fv], 1.0, shade_of(nrmv)) * np.where(W.GLOW[fv], 1.0, light))[:, None]
    fog = np.clip((dist - cfg['fog'][0]) / (cfg['fog'][1] - cfg['fog'][0]), 0, 1)[:, None]
    fog = np.where(W.EMIT[fv][:, None], fog * 0.4, fog)
    col = np.clip(col * (1 - fog) + bot[None, :] * fog, 0, 255).astype(int)
    order = np.argsort(-(dist - W.BIAS[fv]))
    ol = np.where((W.S[fv] >= 0.5)[:, None], col * 0.86, col).astype(int)
    xy = np.stack([sx, sy], -1)[order].round(1).tolist(); cl = col[order].tolist(); oc = ol[order].tolist()
    for q, c, o in zip(xy, cl, oc):
        dr.polygon([tuple(p) for p in q], fill=tuple(c), outline=tuple(o))
    return np.asarray(img).astype(np.float32) * G['vig']


def cam_flow(cam, t, dt):
    """Approximate screen flow (deg/s) of the camera around t: view turn + translation parallax + roll + zoom."""
    p0, l0, r0, f0 = cam.at(t - dt / 2); p1, l1, r1, f1 = cam.at(t + dt / 2)
    d0, d1 = nrm(l0 - p0), nrm(l1 - p1)
    turn = math.degrees(math.acos(np.clip(d0 @ d1, -1, 1)))
    par = math.degrees(math.atan(np.linalg.norm(p1 - p0) / max(np.linalg.norm(l0 - p0), 1e-3)))
    return (turn + 0.5 * par + 0.5 * abs(r1 - r0) + abs(f1 - f0)) / dt


def _render_frame(fi):
    cfg = G['cfg']; t = cfg['t0'] + fi / cfg['fps']
    mb = cfg.get('mblur')            # (max_subframes, shutter 0..1, threshold deg/s)
    if mb:
        K, shut, thr = mb; fl = cam_flow(G['cam'], t, 1 / cfg['fps'])
        n = int(min(K, 1 + fl // thr)) if fl > thr else 1
    else:
        n = 1
    if n <= 1: arr = _render_core(t)
    else:
        arr = sum(_render_core(t + (k / (n - 1) - 0.5) * shut / cfg['fps']) for k in range(n)) / n
    fade = 1.0
    if cfg['fade_in'] > 0: fade = min(fade, (t - cfg['t0']) / cfg['fade_in'])
    if cfg['fade_out'] > 0: fade = min(fade, (cfg['t1'] - t) / cfg['fade_out'])
    arr = arr * np.clip(fade, 0, 1)
    return np.clip(arr, 0, 255).astype(np.uint8).tobytes()


def render(W, cam, sky, out, audio=None, t0=0.0, t1=10.0, fps=30, size=1080, workers=None, near=0.15,
           far=90, fog=(35, 85), pop=0.16, fade_in=0.3, fade_out=0.5, audio_offset=0.0, crf=18, frames=None, sun=None,
           mblur=(7, 0.5, 60.0)):
    """frames=[...] -> preview PNGs '{out}_{fi:05d}.png'; else mp4 (+audio from audio_offset+t0).
    sun(t) -> (direction, angular radius deg, color, halo) or None. Long views: far=150, fog=(50, 140).
    mblur=(max_subframes, shutter, threshold_deg_s): frames whose camera flow exceeds the threshold are averaged over
    several sub-frames (motion blur); slow frames render once. None disables it."""
    yy, xx = np.mgrid[0:size, 0:size] / size - 0.5
    vig = (1 - 0.35 * (xx ** 2 + yy ** 2) ** 1.2 * 2.2)[:, :, None].astype(np.float32)
    G.update(W=W, cam=cam, sky=sky, vig=vig, cfg=dict(t0=t0, t1=t1, fps=fps, size=size, near=near, far=far, fog=fog,
                                                         pop=pop, fade_in=fade_in, fade_out=fade_out, sun=sun, mblur=mblur))
    import multiprocessing as mp
    if frames is not None:
        with mp.get_context('fork').Pool(workers or os.cpu_count()) as pool:
            res = pool.map(_render_frame, frames)
        for fi, fr in zip(frames, res):
            Image.frombytes('RGB', (size, size), fr).save(f'{out}_{fi:05d}.png')
        return
    idx = list(range(int(round((t1 - t0) * fps))))
    cmd = ['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{size}x{size}', '-r', str(fps), '-i', '-']
    if audio: cmd += ['-ss', str(audio_offset + t0), '-t', str(t1 - t0), '-i', audio, '-map', '0:v', '-map', '1:a',
                      '-af', f'afade=t=out:st={max(0, t1 - t0 - fade_out)}:d={fade_out}', '-c:a', 'aac', '-b:a', '192k']
    cmd += ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', str(crf), '-pix_fmt', 'yuv420p', '-shortest', out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with mp.get_context('fork').Pool(workers or os.cpu_count()) as pool:
        for n, fr in enumerate(pool.imap(_render_frame, idx, chunksize=4)):
            p.stdin.write(fr)
            if n % 120 == 0: print(f'[vox2] frame {n}/{len(idx)}', flush=True)
    p.stdin.close(); p.wait()
    print('[vox2] wrote', out)
