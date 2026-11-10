"""story_template.py - runnable BASELINE that only proves the pipeline works (timing, typing, rail camera, render,
audio). Each line is split into short phrases; each phrase is laid out ONCE as a row placed from the camera and its words
type in order inside that row (at most two rows on screen). One hero word per line is typed onto an overhead sign the
rail is timed to pass under. The camera makes one smooth move per line: one framing per line, no per-word stops,
stabilised. Signs are NOT the style to ship: replace every build_line() with the approved correspondence table (SKILL.md step 2).
Usage: python3 story.py --words words.json --audio song.mp3 --t1 28 --out out.mp4 [--preview 1.0,5.2,... --size 540]
       python3 story.py --camcheck        (motion limits, reading rules, stall and handheld checks; no render)"""
import sys, os, json, math, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, 'engine'), HERE]   # works as mv/story.py or inside engine/
from vox import *
from kit import *
from vox2 import *          # vox2 overrides World/render/text3d (move anim, groups, bias, sun, rotation, motion blur)
from layout import *        # typing, text on paths, phrase rows, particles
from rail import Rail, readcheck, stallcheck, steadycheck

ap = argparse.ArgumentParser()
ap.add_argument('--words', default='words.json'); ap.add_argument('--audio', default=None)
ap.add_argument('--t0', type=float, default=0.0); ap.add_argument('--t1', type=float, default=None)
ap.add_argument('--out', default='out.mp4'); ap.add_argument('--size', type=int, default=1080)
ap.add_argument('--preview', default=''); ap.add_argument('--camcheck', action='store_true')
a = ap.parse_args()
WJ = json.load(open(a.words)); t1 = a.t1 or WJ['clip_end']
LINES = [L for L in WJ['lines'] if L['start'] < t1]
OVERRIDE = {}                    # {(line_idx, word_idx): seconds} - manual timing fixes after inspection
LEAD = 0.04                      # typing starts a hair before the syllable
RATE = 14.0                      # letters per second (reference 10-15)
def wt(li, wi): return OVERRIDE.get((li, wi), LINES[li]['words'][wi]['t']) - LEAD
def first_t(li): return min(wt(li, j) for j in range(len(LINES[li]['words']))) if li < len(LINES) else t1 + 1
def next_t(li, wi):
    """Onset of the word after (li, wi), or the clip end."""
    if wi + 1 < len(LINES[li]['words']): return wt(li, wi + 1)
    return min(first_t(li + 1), t1)
def typed(li, wi):
    """Per-letter appear times: syllable chunks typed at RATE; a held syllable stays half-typed until it is sung."""
    w = LINES[li]['words'][wi]['w']; t0 = wt(li, wi); tn = max(next_t(li, wi), t0 + 0.12)
    return type_letters(w, syllable_times(w, t0, tn), RATE, t_next=tn)
def hero_index(li):
    """One hero per line: the longest-held word of 3+ letters (stress / long note), else the last word."""
    ws = LINES[li]['words']; cand = [j for j in range(len(ws)) if len(ws[j]['w'].strip(",.!?'")) >= 3]
    return max(cand, key=lambda j: next_t(li, j) - wt(li, j)) if cand else len(ws) - 1
def phrase_groups(li, skip=(), gap=0.85, max_words=3, max_chars=14):
    """Split a line into short phrases: at a breath (onset gap > gap s), after max_words words or max_chars letters.
    A default only - regroup by hand from the lyrics' sense when writing the storyboard."""
    out, cur = [], []
    for j, w in enumerate(LINES[li]['words']):
        if j in skip: continue
        if cur and (wt(li, j) - wt(li, cur[-1]) > gap or len(cur) >= max_words or
                    sum(len(LINES[li]['words'][k]['w']) for k in cur) + len(w['w']) > max_chars):
            out.append(cur); cur = []
        cur.append(j)
    return out + ([cur] if cur else [])

W = World(seed=11); GAP = 16; SIDE = [(-1) ** i for i in range(len(LINES))]
PAPER = ['#fff4b0', '#ffd6e7', '#cfeeff', '#d9f7c8']; INK = '#2b2b2b'
FONT = 'Fredoka[wdth,wght].ttf'
grass = ('#5fa83a', '#6bb544', '#589f35', '#73bd4b')
def cx(i): return 2.5 * math.sin(i * 1.1)                     # the route bends gently from line to line
floor(W, -16, -GAP * len(LINES) - 16, 16, 34, 0,
      lambda x, z: ('#a7a39a' if abs(x - cx(-z / GAP)) <= 1.2 else grass[(x * 31 + z * 17) % 4]))
for i in range(-1, len(LINES) + 1):
    for sx in (-9, 9): tree(W, sx + (i % 2), 0, -GAP * i + 4, 'cherry' if i % 2 else 'oak')
clouds(W, 20, -50, 50, -GAP * len(LINES) - 40, 30, y=24)

def build_line(i, L, zc):
    """DEFAULT: the hero word typed on an overhead sign that the rail passes under. Replace per line with a literal pun
    scene, and pick a layout (SKILL.md section 6: arc / ring / helix / sphere / road stack / phrase row ...)."""
    h = hero_index(i); w = L['words'][h]['w']; bx = cx(i)
    ht = 1.9; bw = max(8, round(0.62 * ht * len(w) + 1.6))       # hero 10-40% of frame height: grow the sign, not shrink the word
    board(W, (bx, 7.6, zc), (0, 0, 1), (0, 1, 0), bw, 3.2, col=PAPER[i % 4], frame='#6b4423', s=0.5, post=False)
    for px_ in (bx - bw / 2 - 0.25, bx + bw / 2 + 0.25):            # gantry posts well outside the rail
        for y in np.arange(0.75, 9.3, 0.5): W.add((px_, y, zc), 0.5, '#6b4423')
    word_flat(W, w, (bx, 7.6, zc + 0.3), (0, 0, 1), (0, 1, 0), tier='hero', line=i, height=ht, col='#d6336c',
              outline='#3a1020', font_name=FONT, px=16, wght=800, letter_t=typed(i, h), anim='none', bias=0.3)
    return (bx, 7.6, zc + 0.3)

heros = [build_line(i, L, -GAP * i) for i, L in enumerate(LINES)]
way = [(0, 5.5, 26)]                                            # far enough back that the intro keeps moving
for i in range(len(LINES)):
    way += [(cx(i) + SIDE[i] * 0.8, 4.6, -GAP * i + 8.5), (cx(i) - SIDE[i] * 0.3, 4.3, -GAP * i + 2)]   # read, then pass under the sign
way.append((cx(len(LINES)), 4.3, -GAP * len(LINES) + 9))
rail = Rail(way, a.t0, t1 + 0.5, look_ahead=7)
def at_pose(t, p, pose):                     # reading distance = from the designed reading pose to the word
    return (t, p, float(np.linalg.norm(np.asarray(pose, float) - np.asarray(p, float))))
rail.retime([at_pose(max(typed(i, hero_index(i))), heros[i], way[1 + 2 * i]) for i in range(len(LINES))], dwell=None)
rail.aim([(max(typed(i, hero_index(i))), heros[i]) for i in range(len(LINES))], w=0.9, lead=0.6, hold=0.8, swing=0.9)
rail.steady(0.4)                             # dolly, not handheld; placed rows below use the smoothed camera

def visible_later(pos, t, span=4.0):
    for u in np.arange(t, min(t1, t + span), 0.1):
        q = rail.project(u, pos)
        if q is not None and abs(q[0]) < 1.0 and abs(q[1]) < 1.0: return True
    return False

# body words: each phrase is laid out once as a row, placed from the camera at its middle time; words type inside it
ROWS = [-0.28, -0.44]                        # two rows below the hero (the overhead sign leaves through the top)
for i, L in enumerate(LINES):
    h = hero_index(i); groups = phrase_groups(i, skip=(h,))
    for g, idx in enumerate(groups):
        lts = [typed(i, j) for j in idx]; tm = (lts[0][0] + lts[-1][-1]) / 2
        A, N, U = rail.place(tm, 0.0, ROWS[g % 2], 13.0, tilt_to_travel=0.3)   # far enough not to be run into
        ends = [first_t(i + 1)]                                    # exit: the next line starts ...
        if g + 2 < len(groups): ends.append(wt(i, groups[g + 2][0]))   # ... or the row after next reuses this row
        if wt(i, h) > lts[-1][-1]: ends.append(wt(i, h))           # ... or the hero takes the stage
        items = []
        for j, lt in zip(idx, lts):
            w = L['words'][j]['w']; te = max(min(ends), lt[-1] + 0.5)   # never before 0.5 s of reading
            items.append((w, lt, untype_letters(w, te) if te < t1 and visible_later(A, te) else None))
        phrase_row(W, items, A, N, U, height=1.35, gap=0.45, line=i, col=INK, outline='#ffffff', font_name=FONT, px=14,
                   wght=700, anim='none')

W.finalize()
sky = Sky([(0, '#5d9eff', '#d4ecff', 0, 1.0), (t1, '#4a7fe0', '#ffd9c2', 0, 0.95)])
if a.camcheck:
    from camcheck import report
    report(rail, t1, {'lines': LINES}, t0=a.t0); readcheck(rail, W.words, t1); stallcheck(rail, t1, t0=a.t0)
    steadycheck(rail, t1, t0=a.t0); sys.exit()
if a.preview:
    render(W, rail, sky, os.path.splitext(a.out)[0] + '_prev', t0=a.t0, t1=t1, size=a.size if a.size != 1080 else 720,
           frames=[int((float(x) - a.t0) * 30) for x in a.preview.split(',')])
else:
    render(W, rail, sky, a.out, audio=a.audio, t0=a.t0, t1=t1, size=a.size, fade_out=0.3)
