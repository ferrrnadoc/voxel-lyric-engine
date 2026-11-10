#!/usr/bin/env python3
"""segment_scan.py - find the best lyric segment of a song for a voxel one-shot MV.
Input: words.json of the WHOLE song (python3 lrc_words.py song.lrc song.mp3 --out words_full.json).
Prints a line table (section guess, density, repeats) and ranked candidate windows.
Usage: python3 segment_scan.py words_full.json [--min 20 --max 50 --top 6]
Scoring only covers what can be measured (novelty, word density, setup->payoff, clean cut points).
'Drawability' (concrete nouns / literal-able metaphors) must still be judged by reading the lines."""
import json, re, argparse, itertools

def norm(s): return re.sub(r'[^a-z0-9가-힣一-鿿 ]', '', s.lower()).strip()

def jacc(a, b):
    A, B = set(a.split()), set(b.split())
    return len(A & B) / max(1, len(A | B))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('words')
    ap.add_argument('--min', type=float, default=20); ap.add_argument('--max', type=float, default=50)
    ap.add_argument('--top', type=int, default=6)
    ap.add_argument('--draw', default='', help='per-line drawability you judged, e.g. "10:3,11:3,12:2" (0-3, default 1.5)')
    a = ap.parse_args()
    L = [l for l in json.load(open(a.words))['lines'] if l['words']]
    texts = [norm(l['text']) for l in L]
    rep = []
    for i, t in enumerate(texts):                       # repeated lines -> chorus / hook candidates
        rep.append(sum(1 for j, u in enumerate(texts) if j != i and jacc(t, u) >= 0.7))
    print(f"{'#':>3} {'start':>6} {'dur':>5} {'w/s':>4} {'rep':>3} {'gap':>4}  text")
    rows = []
    for i, l in enumerate(L):
        ws = [w['t'] for w in l['words']]
        sung = max(0.6, ws[-1] - ws[0] + 0.45)
        wps = len(ws) / sung
        gap = l['start'] - (L[i - 1]['words'][-1]['t'] + 0.5) if i else l['start']
        rows.append(dict(i=i, start=l['start'], end=l['end'], wps=wps, rep=rep[i], gap=gap))
        tag = 'HOOK' if rep[i] >= 1 else ''
        print(f"{i:>3} {l['start']:6.1f} {l['end'] - l['start']:5.1f} {wps:4.1f} {tag:>4} {gap:4.1f}  {l['text'][:60]}")
    draw = {int(k): float(v) for k, v in (x.split(':') for x in a.draw.split(',') if x)}
    hook = lambda r: r['rep'] >= 1
    cands = []
    for s, e in itertools.combinations(range(len(L) + 1), 2):
        seg = rows[s:e]; t0 = max(0.0, seg[0]['start'] - 0.3); t1 = seg[-1]['end']
        if not (a.min <= t1 - t0 <= a.max): continue
        seen, novelty = set(), 0.0
        for r in seg:                                   # repeats inside the window add little new imagery
            k = texts[r['i']]; again = any(jacc(k, x) >= 0.7 for x in seen)
            novelty += 0.2 if again else (0.8 if hook(r) else 1.0); seen.add(k)
        drawn = sum(draw.get(r['i'], 1.5) for r in seg) / len(seg) / 1.5
        density_pen = sum(max(0, r['wps'] - 3.2) for r in seg) * 0.6        # too fast to read word by word
        setup = sum(1 for r in seg if not hook(r))
        hook_in = 1.5 if setup >= 2 and not hook(seg[0]) and any(hook(r) for r in seg) else 0.0   # setup -> payoff inside
        cliff = 1.0 if e < len(L) and hook(rows[e]) and not hook(seg[-1]) else 0.0                # ends as the hook arrives
        clean = 0.5 if (e == len(L) or rows[e]['gap'] > 0.8 or hook(rows[e]) != hook(seg[-1])) else 0.0   # cut at a seam
        intro = 0.5 if s == 0 else 0.0                                                             # song start = natural opening
        score = novelty / len(seg) * 3 * drawn + hook_in + cliff + clean + intro - density_pen
        cands.append((score, t0, t1, s, e - 1, cliff > 0, hook_in > 0))
    cands.sort(reverse=True)
    print('\nRanked windows (score, clip start-end, lines, ends-at-hook, contains-hook):')
    shown = []
    for c in cands:
        if any(abs(c[1] - x[1]) < 4 and abs(c[2] - x[2]) < 4 for x in shown): continue
        shown.append(c)
        print(f"  {c[0]:5.2f}  {c[1]:6.1f}-{c[2]:6.1f}s  ({c[2] - c[1]:4.1f}s)  lines {c[3]}-{c[4]}  "
              f"{'ends@hook' if c[5] else ''} {'has hook' if c[6] else ''}")
        if len(shown) >= a.top: break

if __name__ == '__main__':
    main()
