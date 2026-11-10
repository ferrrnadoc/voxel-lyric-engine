"""camcheck.py - measure camera motion and flag 'violent' moves (limits calibrated by a viewer, see SKILL.md section 2).
Use from a story:  if a.camcheck: from camcheck import report; report(cam, T1, WJ); sys.exit()
screen flow (deg/s) = how far the picture content sweeps per second, in degrees of view (60 deg/s at fov 60 =
one frame width per second). Measured by projecting a 5x5 grid of points on the plane at the look-target
distance from frame k into frame k+1 (rotation, translation parallax, dolly and zoom all count).
event = contiguous run with flow > 60 deg/s (the fast part of a move), padded 0.1 s each side.
VIOLENT if: peak flow > 250 deg/s, or view direction turns > 90 deg within the event, or peak turn rate > 400 deg/s."""
import math, numpy as np

LIM_FLOW, LIM_TURN, LIM_RATE, FAST = 250.0, 90.0, 400.0, 60.0


def measure(cam, t1, fps=30, t0=0.0):
    dt = 1 / fps; ts = np.arange(t0, t1, dt)
    P, L, F = [], [], []
    for t in ts:
        p, l, r, f = cam.at(t); P.append(p); L.append(l); F.append(f)
    P, L, F = np.array(P, float), np.array(L, float), np.array(F, float)
    D = L - P; dist = np.linalg.norm(D, axis=1); Dn = D / dist[:, None]
    g = np.linspace(-0.8, 0.8, 5); GX, GY = [a.ravel() for a in np.meshgrid(g, g)]
    def basis(k):
        f = Dn[k]; up = (0, 1, 0) if abs(f[1]) < 0.995 else (0, 0, 1)
        r = np.cross(f, up); r /= np.linalg.norm(r); return f, r, np.cross(r, f)
    flow = np.zeros(len(ts) - 1); rate = np.zeros(len(ts) - 1)
    for k in range(len(ts) - 1):
        f, r, u = basis(k); th = math.tan(math.radians(F[k] / 2))
        pts = P[k] + (f[None] + GX[:, None] * th * r[None] + GY[:, None] * th * u[None]) * dist[k]
        f2, r2, u2 = basis(k + 1); th2 = math.tan(math.radians(F[k + 1] / 2))
        Q = pts - P[k + 1]; z = np.maximum(Q @ f2, 1e-3)
        sx, sy = (Q @ r2) / z / th2, (Q @ u2) / z / th2
        flow[k] = np.mean(np.minimum(np.hypot(sx - GX, sy - GY), 2.0)) * (F[k] / 2) / dt
        rate[k] = math.degrees(math.acos(np.clip(Dn[k] @ Dn[k + 1], -1, 1))) / dt
    return ts, Dn, flow, rate


def events(ts, Dn, flow, rate):
    fast = flow > FAST; ev = []; i = 0
    while i < len(fast):
        if not fast[i]: i += 1; continue
        j = i
        while j < len(fast) and fast[j]: j += 1
        a, b = max(0, i - 3), min(len(fast) - 1, j + 3)
        turn = math.degrees(math.acos(np.clip(Dn[a] @ Dn[b], -1, 1)))
        e = dict(t0=ts[a], t1=ts[b], peak=flow[a:b].max(), turn=turn, rate=rate[a:b].max())
        e['violent'] = e['peak'] > LIM_FLOW or e['turn'] > LIM_TURN or e['rate'] > LIM_RATE
        ev.append(e); i = j
    return ev


def report(cam, t1, WJ=None, t0=0.0):
    ts, Dn, flow, rate = measure(cam, t1, t0=t0)
    ev = events(ts, Dn, flow, rate)
    starts = [L['words'][0]['t'] for L in WJ['lines'] if L['words']] if WJ else []
    line_of = lambda t: max([n for n, s in enumerate(starts) if s - 0.5 <= t] or [-1])
    print(f'limits: peak flow <= {LIM_FLOW:.0f} deg/s, turn <= {LIM_TURN:.0f} deg per move, turn rate <= {LIM_RATE:.0f} deg/s')
    print(' line   t0-t1        peak  turn  rate')
    for e in ev:
        flag = '  <-- VIOLENT' if e['violent'] else ''
        print(f" L{line_of(e['t1']):<4}{e['t0']:5.2f}-{e['t1']:5.2f}  {e['peak']:5.0f} {e['turn']:5.0f} {e['rate']:5.0f}{flag}")
    calm = (flow < 15).mean() * 100
    print(f'violent events: {sum(e["violent"] for e in ev)} / {len(ev)} fast events;  calm (<15 deg/s) {calm:.0f}% of frames')
    return ev, ts, flow
