"""layout.py - lyric typography for vox2: text that follows geometry, syllable typing / un-typing, emotion particles.

Text layouts (all letters are voxel cubes rotated into the letter's own frame, see vox2 World.add(M=, gk=)):
  ArcPath     - letters on a circle around an axis: wrap a tower top, curve over a clock face, orbit a head (ring)
  ArcPath(pitch>0) - helix / spiral (tunnel walls, rising staircases)
  PolyPath    - any smooth 3D curve through points: road, river, flight trail, thrown-object arc; lie=True lays the
                letters flat on the ground along the curve
  SpherePath  - along a latitude of a sphere, letters follow the surface normal (printed on a globe, a ball, a balloon)
Frame convention: R = reading direction (tangent), U = letter up, N = facing (towards the reader), R = U x N.

Typing (the reference's default reveal): letters appear one by one at 10-15 letters/s, in sung-syllable chunks;
a held syllable stays half-typed until the next syllable is sung. Exits un-type in reading order.
"""
import math, numpy as np
from vox import nrm, hexc
import vox2

# ----------------------------------------------------------------------------- syllables & typing schedules
_PY = None
def syllables(word):
    """Split a word into sung-syllable letter chunks (pyphen if installed, else vowel-group heuristic)."""
    global _PY
    core = word
    try:
        import pyphen
        if _PY is None: _PY = pyphen.Pyphen(lang='en_US', left=1, right=2)
        parts = _PY.inserted(core, hyphen='\x00').split('\x00')
        if len(parts) > 1: return parts
    except Exception:
        pass
    v = 'aeiouyAEIOUY'; idx = [i for i, ch in enumerate(core) if ch in v and (i == 0 or core[i - 1] not in v)]
    if core.lower().endswith('e') and len(idx) > 1 and idx[-1] == len(core) - 1: idx = idx[:-1]
    if len(idx) <= 1: return [core]
    cuts = []
    for a, b in zip(idx, idx[1:]):
        j = a + 1
        while j < len(core) and core[j] in v: j += 1
        cons = b - j
        cuts.append(b - 1 if cons >= 1 else b)
    out, last = [], 0
    for c in cuts: out.append(core[last:c]); last = c
    out.append(core[last:]); return [p for p in out if p]


def syllable_times(word, t0, t1, onsets=None, per=0.28):
    """Start time of each syllable chunk of `word`, sung from t0 (word onset) until at most t1 (next word onset).
    onsets: optional vocal onset / pitch-segment start times (from lrc_words.vocal_onsets or pitchseg) - the first
    chunk starts at t0, later chunks snap to the nearest onset inside the window; otherwise ~per seconds apart."""
    ch = syllables(word); n = len(ch)
    if n == 1: return [t0]
    span = min(t1 - t0, per * n * 1.3)
    prior = [t0 + span * k / n for k in range(n)]
    if onsets is not None:
        cand = [o for o in onsets if t0 + 0.08 < o < t1 - 0.05]
        out = [t0]
        for k in range(1, n):
            best = min([c for c in cand if c > out[-1] + 0.07] or [prior[k]], key=lambda c: abs(c - prior[k]))
            out.append(max(best, out[-1] + 0.07))
        return out
    return prior


def type_letters(word, syl_t, rate=14.0, t_next=None, max_frac=0.8):
    """Per-character appear times: each syllable chunk types at `rate` letters/s from its start time; a chunk is
    compressed so it finishes within max_frac of the gap to the next chunk. Non-letters ride with their neighbour."""
    ch = syllables(word); out = []
    ends = list(syl_t[1:]) + [t_next if t_next is not None else syl_t[-1] + 0.6]
    for c, t0, t1 in zip(ch, syl_t, ends):
        step = min(1.0 / rate, max(0.02, max_frac * (t1 - t0) / max(1, len(c))))
        out += [t0 + i * step for i in range(len(c))]
    return out[:len(word)] + [out[-1]] * (len(word) - len(out))


def untype_letters(word, t0, rate=16.0):
    """Per-character disappear times for un-typing in reading order starting at t0."""
    return [t0 + i / rate for i in range(len(word))]


# ----------------------------------------------------------------------------- paths
class ArcPath:
    """Circle (or helix if pitch != 0) around `axis` through `center`. facing='out' reads from outside (wrap a tower,
    ring around a character), 'in' from inside (tunnel walls, inside a dome). start_deg: where the text starts,
    measured from `ref` (a vector perpendicular to axis). pitch: axial rise per full turn (helix).
    lean (deg) + lean_to (vector): tilt each letter about its baseline toward where the camera comes from, e.g. tunnel
    wall text leaning toward the entrance so it is read at ~30-45 deg instead of edge-on."""
    def __init__(self, center, axis, radius, start_deg=0.0, facing='out', ref=None, pitch=0.0, lean=0.0, lean_to=None):
        self.c = np.asarray(center, float); self.a = nrm(axis); self.r = float(radius); self.p = float(pitch)
        ref = np.asarray(ref if ref is not None else ((1, 0, 0) if abs(self.a[0]) < 0.9 else (0, 0, 1)), float)
        self.e1 = nrm(ref - (ref @ self.a) * self.a); self.e2 = np.cross(self.a, self.e1)
        self.th0 = math.radians(start_deg); self.out = facing == 'out'
        self.sign = 1.0 if self.out else -1.0      # direction of travel so that R = U x N reads left->right
        self.length = 2 * math.pi * self.r
        self.lean = math.radians(lean); self.lean_to = None if lean_to is None else nrm(lean_to)

    def frame(self, s):
        th = self.th0 + self.sign * s / self.r
        rad = math.cos(th) * self.e1 + math.sin(th) * self.e2
        P = self.c + rad * self.r + self.a * (self.p * (th - self.th0) / (2 * math.pi))
        tang = (-math.sin(th) * self.e1 + math.cos(th) * self.e2) * self.sign * self.r + self.a * self.sign * self.p / (2 * math.pi)
        R = nrm(tang); N0 = rad if self.out else -rad
        U = nrm(self.a - (self.a @ R) * R); N = np.cross(R, U)
        if N @ N0 < 0: U = -U; N = -N
        if self.lean and self.lean_to is not None:
            l = self.lean if U @ self.lean_to >= 0 else -self.lean
            N, U = math.cos(l) * N + math.sin(l) * U, math.cos(l) * U - math.sin(l) * N
        return P, R, U, N


class PolyPath:
    """Smooth curve through 3D points (centripetal Catmull-Rom). up: letter-up hint (world up by default).
    lie=True lays letters flat on the ground: N = up (facing the sky), letter tops point to the left of travel."""
    def __init__(self, points, up=(0, 1, 0), lie=False, samples=600):
        P = [np.asarray(p, float) for p in points]
        P = [2 * P[0] - P[1]] + P + [2 * P[-1] - P[-2]]
        pts = []
        for i in range(1, len(P) - 2):
            p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
            for u in np.linspace(0, 1, max(4, samples // (len(P) - 3)), endpoint=False):
                pts.append(0.5 * (2 * p1 + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3))
        pts.append(P[-2]); self.pts = np.array(pts)
        seg = np.linalg.norm(np.diff(self.pts, axis=0), axis=1); self.s = np.r_[0, np.cumsum(seg)]
        self.length = float(self.s[-1]); self.up = nrm(up); self.lie = lie

    def frame(self, s):
        s = float(np.clip(s, 0, self.length)); i = int(np.clip(np.searchsorted(self.s, s) - 1, 0, len(self.pts) - 2))
        u = (s - self.s[i]) / max(self.s[i + 1] - self.s[i], 1e-9)
        P = self.pts[i] * (1 - u) + self.pts[i + 1] * u; R = nrm(self.pts[i + 1] - self.pts[i])
        if self.lie:
            N = nrm(self.up - (self.up @ R) * R); U = np.cross(N, R)
        else:
            U = nrm(self.up - (self.up @ R) * R); N = np.cross(R, U)
        return P, R, U, N


class SpherePath:
    """Along latitude lat_deg of a sphere (center, radius); letters lie on the surface (N = outward normal)."""
    def __init__(self, center, radius, lat_deg=0.0, start_deg=0.0, axis=(0, 1, 0), ref=None, lift=0.0):
        self.arc = ArcPath(np.asarray(center, float) + nrm(axis) * radius * math.sin(math.radians(lat_deg)), axis,
                           radius * math.cos(math.radians(lat_deg)), start_deg, 'out', ref)
        self.c = np.asarray(center, float); self.R = radius + lift; self.length = self.arc.length

    def frame(self, s):
        P, R, U, N = self.arc.frame(s)
        n = nrm(P - self.c); P = self.c + n * self.R
        U = nrm(np.cross(n, R)); return P, R, U, n


# ----------------------------------------------------------------------------- text on a path
_LID = [0]
def text_on_path(W, word, path, s0=None, align='center', height=1.0, col='#ffffff', side=None, outline='#1b1b2a',
                 depth=2, font_name='Fredoka[wdth,wght].ttf', px=14, wght=None, thr=100, t_in=-1e9, t_out=1e9,
                 letter_t=None, letter_tout=None, anim='none', emit=False, glow=True, bias=0.0, grp=-1, spacing=1.0,
                 tier='body', line=None, register=True):
    """Lay `word` along `path` (ArcPath / PolyPath / SpherePath). Each letter sits in its own frame (R,U,N) at its
    advance position; letters are rotated cubes (culled per letter). align: where s0 is ('center'|'start').
    letter_t / letter_tout: per-character times (use type_letters / untype_letters). Returns (s_start, s_end)."""
    m, bounds = vox2.glyph_mask(word, font_name, px, wght, thr)
    ring = None
    if outline is not None:
        from scipy.ndimage import binary_dilation
        ring = binary_dilation(m, iterations=1) & ~m
    h, w = m.shape; vs = height / h; total = w * vs * spacing
    if s0 is None: s0 = path.length / 2 if align == 'center' else 0.0
    start = s0 - total / 2 if align == 'center' else s0
    side = side or tuple(int(x * 0.55) for x in hexc(col))
    nchar = len(word)
    for k in range(nchar):
        a = int(round(bounds[k])); b = int(round(bounds[k + 1])) if k + 1 < len(bounds) else w
        a, b = max(0, a), min(w, max(b, a + 1))
        cols = [j for j in range(a, b) if (m[:, j].any() or (ring is not None and ring[:, j].any()))]
        if not cols or word[k] == ' ': continue
        jc = (a + b) / 2.0
        P, R, U, N = path.frame(start + jc * vs * spacing)
        Mrot = np.stack([R, U, N], axis=1); mid = W.mat(Mrot); _LID[0] += 1; lid = _LID[0]
        ti = letter_t[k] if letter_t is not None else t_in
        to = letter_tout[k] if letter_tout is not None else t_out
        for i in range(h):
            for j in cols:
                on = m[i, j]; ol = ring is not None and ring[i, j]
                if not (on or ol): continue
                lx, ly = (j - jc + 0.5) * vs, (h / 2 - i - 0.5) * vs
                base = P + R * lx + U * ly
                if on:
                    for d in range(depth):
                        W.add(base - N * d * vs, vs, col if d == 0 else side, ti, to, P, anim, emit, jitter=0.03,
                              glow=glow, grp=grp, bias=bias, M=mid, gk=(lid, j, -i, -d))
                else:
                    for d in range(depth):
                        W.add(base - N * (d + 0.5) * vs, vs, outline, ti, to, P, anim, False, jitter=0.0, glow=glow,
                              grp=grp, bias=bias, M=mid, gk=(lid + 100000, j, -i, -d))
    if register:
        Pc, R, U, N = path.frame(start + total / 2)
        tf = max(letter_t) if letter_t is not None else t_in
        W.words.append(dict(w=word, t_full=tf, t_out=(min(letter_tout) if letter_tout is not None else t_out),
                            pos=Pc, height=height * 0.6, N=N, U=U, tier=tier, line=line))
    return start, start + total


def word_flat(W, word, pos, N=(0, 0, 1), U=(0, 1, 0), tier='body', line=None, **kw):
    """vox2.text3d + registration for readcheck. Accepts letter_t / letter_tout like text3d."""
    wd = vox2.text3d(W, word, pos, N, U, **kw)
    lt, lo = kw.get('letter_t'), kw.get('letter_tout')
    W.words.append(dict(w=word, t_full=max(lt) if lt else kw.get('t_in', -1e9),
                        t_out=min(lo) if lo else kw.get('t_out', 1e9), pos=np.asarray(pos, float),
                        height=kw.get('height', 1.5) * 0.6, N=nrm(N), U=nrm(U), tier=tier, line=line))
    return wd


def phrase_row(W, items, center, N=(0, 0, 1), U=(0, 1, 0), height=1.0, gap=0.3, line=None, **kw):
    """A phrase laid out ONCE as one row on one plane; its words are then revealed in order inside that fixed layout.
    Nothing moves after layout, so words of one phrase never overlap or hide each other (placing every word on its own
    makes dense lines a scatter). items: [(word, letter_t, letter_tout_or_None, {per-word overrides})].
    Returns (word centres, total width)."""
    N = nrm(N); U = nrm(U); R = np.cross(U, N); ws = []
    for it in items:
        o = dict(kw); o.update(it[3] if len(it) > 3 and it[3] else {}); h = o.get('height', height)
        m, _ = vox2.glyph_mask(it[0], o.get('font_name', 'Fredoka[wdth,wght].ttf'), o.get('px', 14), o.get('wght'))
        ws.append(m.shape[1] * h / m.shape[0])
    total = sum(ws) + gap * (len(ws) - 1); x = -total / 2; out = []
    for it, wdt in zip(items, ws):
        o = dict(kw); o.update(it[3] if len(it) > 3 and it[3] else {}); h = o.pop('height', height)
        pos = np.asarray(center, float) + R * (x + wdt / 2)
        word_flat(W, it[0], pos, N, U, line=line, height=h, letter_t=it[1], letter_tout=it[2], **o)
        out.append(pos); x += wdt + gap
    return out, total


# ----------------------------------------------------------------------------- emotion particles
HEART_S = ['.X.X.', 'XXXXX', '.XXX.', '..X..']
SPARK_S = ['.X.', 'XXX', '.X.']
CROSS_S = ['X.X', '.X.', 'X.X']

def particles(W, center, size, t0, t1, rate=6.0, shapes=(HEART_S,), cols=('#ff7eb3',), vs=0.08, rise=1.5, life=1.6,
              seed=3, glow=True, emit=True):
    """Floating emotion particles (hearts, sparkles, crosses) drifting up inside a box (center, size=(sx,sy,sz)),
    spawned at `rate` per second between t0 and t1, each living `life` seconds. Uses anim='move' + t_out."""
    rng = np.random.default_rng(seed); c = np.asarray(center, float); sz = np.asarray(size, float)
    n = int((t1 - t0) * rate)
    for k in range(n):
        ts = t0 + (t1 - t0) * k / max(1, n) + rng.uniform(0, 1 / rate)
        p = c + (rng.uniform(-0.5, 0.5, 3) * sz); shp = shapes[rng.integers(len(shapes))]; col = cols[rng.integers(len(cols))]
        h, w = len(shp), len(shp[0]); drift = np.array([rng.uniform(-0.3, 0.3), rise, rng.uniform(-0.3, 0.3)])
        for i, row in enumerate(shp):
            for j, ch in enumerate(row):
                if ch == 'X':
                    q = p + np.array([(j - w / 2) * vs, (h / 2 - i) * vs, 0])
                    W.add(q + drift, vs, col, ts, ts + life, q, 'move', emit, jitter=0.05, glow=glow, off=-drift, md=life)


def path_stack(W, words, path, s_start, gap=2.5, height=1.2, lift=0.03, facing='up', line=None, **kw):
    """Road-marking stack: each word lies ACROSS the path (flat, tops pointing forward along the path) and the words
    follow one another down the path, so a camera flying forward reads them in order - motion = reading.
    words: [(word, letter_t[, letter_tout])]. facing='up' lies on the ground; 'back' stands up facing the camera."""
    s = s_start; out = []
    for item in words:
        word, lt = item[0], item[1]; lo = item[2] if len(item) > 2 else None
        P, R, U, N = path.frame(s)
        n_up = np.array([0.0, 1.0, 0.0])
        if facing == 'up': Nw, Uw = n_up, R            # lying flat, letter tops point down the path
        else: Nw, Uw = -R, n_up                        # standing up, facing the oncoming camera
        word_flat(W, word, P + n_up * lift, Nw, Uw, height=height, letter_t=lt, letter_tout=lo, line=line, **kw)
        out.append(s); s += gap
    return out
