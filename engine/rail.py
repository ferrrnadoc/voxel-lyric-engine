"""rail.py - camera that never stops, and lyrics placed so they land in the moving frame.

Why: keyframed cameras hesitate at every key (ease in/out), and words placed independently force the camera to stop
and look at them. Here the camera rides a RAIL (a smooth path + a time->distance curve that is strictly increasing),
and each word is PLACED FROM THE CAMERA: at the moment it finishes typing it sits at a chosen screen point and size,
then the camera's own motion carries it across and out of frame. Motion = reading.

  rail = Rail(waypoints, t0, t1, speed=[(t, v), ...] or anchors=[(t, s), ...], look_ahead=6, targets=[(t, point, w)])
  anchor, N, U = rail.place(t_full, sx, sy, dist)      # sx, sy in [-1, 1] (0,0 = centre, +y up)
  rail.retime([(t, world_point, view_dist), ...])        # objects first: time the camera to pass existing things
  readcheck(rail, W.words); stallcheck(rail, t1)         # numbers to verify the reading rules
"""
import math, numpy as np
from vox import nrm
from layout import PolyPath


class Rail:
    def __init__(self, waypoints, t0, t1, speed=None, anchors=None, look_ahead=6.0, targets=None, roll=None,
                 fov=60.0, vmin_frac=0.35):
        self.path = PolyPath(waypoints, samples=1500); self.L = self.path.length
        self.t0, self.t1 = float(t0), float(t1); self.ahead = float(look_ahead)
        self.targets = sorted(targets or [], key=lambda x: x[0])
        self.roll = sorted(roll or [(t0, 0.0)]); self.fovk = fov if isinstance(fov, list) else [(t0, float(fov))]
        self.vmin_frac = vmin_frac
        if anchors: self.retime_s(anchors)
        else: self.set_speed(speed or [(t0, 1.0), (t1, 1.0)])

    # ---- time -> distance along the rail
    def set_speed(self, keys):
        """Relative speed profile [(t, v)], linear between keys, never below vmin_frac x mean; scaled to cover the rail."""
        ts = np.linspace(self.t0, self.t1, 2000); kt, kv = zip(*sorted(keys))
        v = np.interp(ts, kt, kv); v = np.maximum(v, self.vmin_frac * v.mean())
        s = np.r_[0, np.cumsum((v[1:] + v[:-1]) / 2 * np.diff(ts))]; s *= self.L / s[-1]
        self._ts, self._s = ts, s

    def retime_s(self, anchors):
        """anchors [(t, s)] -> monotone PCHIP s(t). s must strictly increase with t (no stops, no reversing)."""
        from scipy.interpolate import PchipInterpolator
        a = sorted(anchors); ts, ss = zip(*a)
        if any(b <= x for x, b in zip(ss, ss[1:])): raise ValueError(f'rail anchors must strictly increase: {a}')
        f = PchipInterpolator(ts, ss, extrapolate=True)
        self._ts = np.linspace(self.t0, self.t1, 2000); self._s = np.clip(f(self._ts), 0, self.L)

    def retime(self, items, dwell=(0.9, 1.2), creep=None):
        """items [(t, world_point, view_dist)]: pass each point at view_dist in front of the camera at time t.
        creep=[(t_a, t_b, dist)]: windows where the camera only creeps `dist` (for words placed with place()).
        dwell=(seconds, distance): after each word the camera only creeps `distance` further during `seconds`
        (slow drift while it is read), then speeds up to the next one - reading slow, transitions fast, never zero."""
        anchors = []; i0 = 0
        for t, p, d in sorted(items, key=lambda x: x[0]):   # search forward only: anchors follow the time order
            dd = np.linalg.norm(self.path.pts[i0:] - np.asarray(p, float), axis=1)
            i = i0 + int(np.argmin(np.abs(dd - d) + 0.3 * dd * (np.arange(len(dd)) > np.argmin(dd))))
            anchors.append((t, float(self.path.s[i]))); i0 = min(i + 1, len(self.path.pts) - 1)
        if dwell:
            extra = []
            srt = sorted(anchors)
            for k, (t, s_) in enumerate(srt):
                nt, ns = (srt[k + 1] if k + 1 < len(srt) else (self.t1, self.L))
                if t + dwell[0] < nt - 0.25 and s_ + dwell[1] < ns - 0.5: extra.append((t + dwell[0], s_ + dwell[1]))
            anchors += extra
        if not any(abs(t - self.t0) < 1e-6 for t, _ in anchors): anchors.append((self.t0, 0.0))
        if not any(abs(t - self.t1) < 1e-6 for t, _ in anchors): anchors.append((self.t1, self.L))
        self.retime_s(anchors)
        for ta, tb, dist in (creep or []):          # reading windows for words placed from the camera
            sa = self.s_at(ta); anchors = [x for x in anchors if not (ta - 0.05 < x[0] < tb + 0.05)]
            anchors += [(ta, sa), (tb, sa + dist)]
            self.retime_s(anchors)

    def aim(self, items, w=0.8, lead=0.45, hold=0.7, swing=0.6, release=1.2):
        """items [(t, point)]: frame each point from t-lead to t+hold (weight w); look along the rail in long gaps.
        Consecutive words share the gap between them: the view never snaps (at least `swing` seconds per turn when
        the words are that far apart, otherwise the whole gap) and targets never leave time order."""
        items = sorted(items, key=lambda x: x[0]); n = len(items)
        S = [t - lead for t, _ in items]; E = [t + hold for t, _ in items]
        for k in range(n - 1):
            t0, t1 = items[k][0], items[k + 1][0]
            E[k] = min(E[k], t1); S[k + 1] = max(S[k + 1], t0)
            if S[k + 1] - E[k] < swing:                  # not enough time to turn: split the gap
                if t1 - t0 <= swing: E[k], S[k + 1] = t0, t1
                else:
                    mid = (E[k] + S[k + 1]) / 2
                    E[k] = max(t0, mid - swing / 2); S[k + 1] = min(t1, E[k] + swing)
        tg = []
        for k, (t, p) in enumerate(items):
            tg += [(S[k], p, w), (t, p, max(w, 0.97)), (E[k], p, w * 0.75 if E[k] - t > 0.3 else w)]   # exact at the word time
            if k + 1 < n and S[k + 1] - E[k] > release: tg.append(((E[k] + S[k + 1]) / 2, items[k + 1][1], 0.0))
            if k + 1 == n: tg.append((E[k] + 1.0, p, 0.0))     # after the last word: look along the rail again
        self.targets = tg                                       # already in time order

    def steady(self, sigma=0.35, dt=1 / 120):
        """Stabiliser: Gaussian-smooth position, view direction, roll and fov over `sigma` seconds, so the camera moves
        like a dolly, not a handheld camera. Call after retime/aim and BEFORE place(): at/place/project then all use the
        smoothed camera, so placed words still land where they were asked to."""
        self._st = None
        ts = np.arange(self.t0, self.t1 + dt / 2, dt); S = [self.at(t) for t in ts]
        P = np.array([x[0] for x in S]); D = np.array([nrm(x[1] - x[0]) for x in S])
        Rl = np.array([x[2] for x in S], float); F = np.array([x[3] for x in S], float)
        k = max(1, int(round(3 * sigma / dt))); g = np.exp(-0.5 * ((np.arange(-k, k + 1) * dt) / sigma) ** 2); g /= g.sum()
        def sm(a):
            pad = np.concatenate([np.repeat(a[:1], k, 0), a, np.repeat(a[-1:], k, 0)])
            if a.ndim == 1: return np.convolve(pad, g, 'valid')
            return np.stack([np.convolve(pad[:, j], g, 'valid') for j in range(a.shape[1])], 1)
        P, D, Rl, F = sm(P), sm(D), sm(Rl), sm(F); D /= np.linalg.norm(D, axis=1)[:, None]
        self._st = (ts, P, D, Rl, F)
        return self

    def s_at(self, t): return float(np.interp(t, self._ts, self._s))

    # ---- camera interface (same as vox.Camera.at)
    def at(self, t):
        st = getattr(self, '_st', None)
        if st is not None:
            ts, P_, D_, R_, F_ = st; t = float(np.clip(t, ts[0], ts[-1]))
            P = np.array([np.interp(t, ts, P_[:, j]) for j in range(3)]); d = nrm(np.array([np.interp(t, ts, D_[:, j]) for j in range(3)]))
            return P, P + d * 5.0, float(np.interp(t, ts, R_)), float(np.interp(t, ts, F_))
        s = self.s_at(t); P, R, U, N = self.path.frame(s)
        ahead = self.path.frame(min(self.L, s + self.ahead))[0]
        d = nrm(ahead - P) if np.linalg.norm(ahead - P) > 1e-3 else R
        if self.targets:
            tt = [x[0] for x in self.targets]
            k = int(np.clip(np.searchsorted(tt, t), 1, len(tt) - 1)) if len(tt) > 1 else 0
            if len(tt) == 1: tp, w = np.asarray(self.targets[0][1], float), self.targets[0][2]
            else:
                (ta, pa, wa), (tb, pb, wb) = self.targets[k - 1], self.targets[k]
                u = np.clip((t - ta) / max(tb - ta, 1e-6), 0, 1); u = u * u * (3 - 2 * u)
                tp = np.asarray(pa, float) * (1 - u) + np.asarray(pb, float) * u; w = wa * (1 - u) + wb * u
            d = nrm((1 - w) * d + w * nrm(tp - P))
        roll = float(np.interp(t, *zip(*self.roll))); fov = float(np.interp(t, *zip(*self.fovk)))
        return P, P + d * 5.0, roll, fov

    def basis(self, t):
        P, L, roll, fov = self.at(t); f = nrm(L - P); up = (0, 1, 0) if abs(f[1]) < 0.995 else (0, 0, 1)
        r = nrm(np.cross(f, up)); u = np.cross(r, f); a = math.radians(roll)
        r, u = r * math.cos(a) + u * math.sin(a), u * math.cos(a) - r * math.sin(a)
        return P, f, r, u, fov

    def ray(self, t, sx, sy):
        P, f, r, u, fov = self.basis(t); th = math.tan(math.radians(fov) / 2)
        return P, nrm(f + sx * th * r + sy * th * u)

    def place(self, t, sx, sy, dist, face='camera', tilt_to_travel=0.0):
        """World anchor that will be at screen (sx, sy), `dist` away, at time t. Returns (anchor, N, U).
        face='camera': text faces the camera at t. tilt_to_travel (0..1): turn the face toward the oncoming camera
        (later in the rail) so it stays readable while the camera approaches/passes."""
        P, d = self.ray(t, sx, sy); A = P + d * dist; _, f, r, u, _ = self.basis(t)
        N = -d
        if tilt_to_travel:
            P2 = self.at(min(self.t1, t + 0.6))[0]; N = nrm((1 - tilt_to_travel) * N + tilt_to_travel * nrm(P2 - A))
        U = nrm(u - (u @ N) * N); return A, N, U

    def project(self, t, p):
        P, f, r, u, fov = self.basis(t); q = np.asarray(p, float) - P; z = q @ f; th = math.tan(math.radians(fov) / 2)
        if z <= 1e-3: return None
        return (q @ r) / z / th, (q @ u) / z / th, z


def readcheck(rail, words, t_end=None, fps=30):
    """Per word: where it is when fully typed (screen x,y in -1..1), how big (fraction of frame height), how oblique,
    and how long it stays readable (inside the frame and before its exit time). Flags rule violations."""
    t_end = t_end or rail.t1; rows = []
    print(' word            t_full   x     y    size  angle  dwell  flags')
    for w in sorted(words, key=lambda d: d['t_full']):
        tf = w['t_full']
        if not (rail.t0 <= tf <= t_end): continue
        pr = rail.project(tf, w['pos']); fl = []
        if pr is None: print(f" {w['w'][:14]:14s} {tf:6.2f}   behind camera  <-- BEHIND"); continue
        sx, sy, z = pr; th = math.tan(math.radians(rail.at(tf)[3]) / 2); size = w['height'] / (2 * z * th)
        P = rail.at(tf)[0]; ang = math.degrees(math.acos(np.clip(np.asarray(w['N']) @ nrm(P - w['pos']), -1, 1)))
        dwell = 0.0; t = tf
        while t < min(t_end, w['t_out']):
            q = rail.project(t, w['pos'])
            if q is None or abs(q[0]) > 1.0 or abs(q[1]) > 1.0: break
            t += 1 / fps; dwell += 1 / fps
        hero = w.get('tier') == 'hero'
        if abs(sx) > 0.4: fl.append('OFF-CENTRE')
        if size < (0.10 if hero else 0.04): fl.append('SMALL')
        if not hero and size > 0.12: fl.append('BIG')
        lying = abs(np.asarray(w['N'], float) @ np.array([0.0, 1.0, 0.0])) > 0.8   # ground text is read foreshortened
        if ang > (70 if lying else 55): fl.append('OBLIQUE')
        if dwell < 0.5: fl.append('SHORT')
        rows.append(dict(w=w['w'], t=tf, x=sx, y=sy, size=size, ang=ang, dwell=dwell, flags=fl))
        print(f" {w['w'][:14]:14s} {tf:6.2f} {sx:+.2f} {sy:+.2f}  {size * 100:4.1f}%  {ang:4.0f}°  {dwell:4.2f}s  {' '.join(fl)}")
    if rows:
        d = np.array([r['dwell'] for r in rows])
        print(f'median dwell {np.median(d):.2f}s (reference ~1.2s, range 0.5-2.5); flagged {sum(bool(r["flags"]) for r in rows)}/{len(rows)}')
    return rows


def stallcheck(cam, t1, t0=0.0, fps=30):
    """The camera should never stop: screen flow < 4 deg/s only in short runs (reference: 3.4% of frames, longest
    0.73 s; median flow ~18 deg/s). Uses camcheck.measure."""
    from camcheck import measure
    ts, Dn, flow, rate = measure(cam, t1, fps, t0)
    stall = flow < 4.0; runs, cur, longest = [], 0, 0
    for v in stall:
        cur = cur + 1 if v else 0; longest = max(longest, cur)
    print(f'flow p25/p50/p75/p95 {np.percentile(flow, 25):.0f}/{np.percentile(flow, 50):.0f}/{np.percentile(flow, 75):.0f}/{np.percentile(flow, 95):.0f} deg/s (reference 9/18/27/42), peak {flow.max():.0f}')
    print(f'flow median {np.median(flow):.1f} deg/s (reference ~18); <4 deg/s {stall.mean() * 100:.1f}% of frames '
          f'(reference 3.4%), longest {longest / fps:.2f}s (reference 0.73s){"  <-- STALL" if longest / fps > 0.75 else ""}')
    return flow


def steadycheck(cam, t1, t0=0.0, fps=30):
    """Handheld feel = the picture's motion keeps changing direction. Tracks the screen motion of the look point
    (what optical flow sees at the image centre) and counts pan/tilt reversals and sharp heading changes.
    Reference (Lover Girl, optical flow): 6.9 reversals per 10 s, 9% of 1/3-s steps turn > 45 deg."""
    ts = np.arange(t0, t1, 1 / fps); dx, dy = [], []
    for a, b in zip(ts, ts[1:]):
        P, L, r, f = cam.at(a); q = cam.project(b, L) if hasattr(cam, 'project') else None
        if q is None: dx.append(0.0); dy.append(0.0); continue
        dx.append(q[0] * 270); dy.append(-q[1] * 270)       # pixels per frame at 540 px, like the optical-flow pass
    dx, dy = np.array(dx), np.array(dy)
    def sm(x, k): return np.convolve(x, np.ones(k) / k, 'same')
    def rev(x, thr=1.0):
        s_, c = 0, 0
        for v in x:
            if abs(v) < thr: continue
            g = 1 if v > 0 else -1; c += bool(s_ and g != s_); s_ = g
        return c
    T = len(dx) / fps; r10 = (rev(sm(dx, 5)) + rev(sm(dy, 5))) / T * 10
    m = fps // 3; seg = [(dx[i:i + m].mean(), dy[i:i + m].mean()) for i in range(0, len(dx) - m, m)]
    ang = [abs(math.degrees(math.atan2(a1 * b2 - b1 * a2, a1 * a2 + b1 * b2))) for (a1, b1), (a2, b2) in zip(seg, seg[1:])
           if math.hypot(a1, b1) > 1.5 and math.hypot(a2, b2) > 1.5]
    sharp = 100 * np.mean(np.array(ang) > 45) if ang else 0.0
    flag = '  <-- HANDHELD' if r10 > 10 or sharp > 15 else ''
    print(f'pan/tilt reversals {r10:.1f} per 10 s (reference 6.9); heading change > 45 deg in {sharp:.0f}% of 1/3-s steps '
          f'(reference 9%){flag}')
    return r10, sharp
