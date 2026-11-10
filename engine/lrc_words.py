#!/usr/bin/env python3
"""LRC + audio -> per-word sung times (words.json).
Usage: python3 lrc_words.py song.lrc song.mp3 --end 28 --out words.json
- Standard LRC ([mm:ss.xx] line) gives line starts; word times are estimated by snapping
  to vocal onsets (REPET-SIM foreground) with a syllable-proportional prior (DP alignment).
- Enhanced LRC (<mm:ss.xx>word) word tags are used as-is (most accurate)."""
import re, sys, json, argparse, numpy as np, librosa

def parse_lrc(path):
    lines = []
    for raw in open(path, encoding='utf-8-sig'):
        tags = re.findall(r'\[(\d+):(\d+(?:\.\d+)?)\]', raw)
        text = re.sub(r'\[[^\]]*\]', '', raw).strip()
        for m, s in tags:
            lines.append((int(m) * 60 + float(s), text))
    lines.sort(key=lambda x: x[0])
    return lines

def syllables(w):
    w = re.sub(r'[^a-z]', '', w.lower())
    if not w: return 1
    n = len(re.findall(r'[aeiouy]+', w))
    if w.endswith('e') and n > 1 and not w.endswith('le'): n -= 1
    return max(1, n)

def vocal_onsets(y, sr):
    S, ph = librosa.magphase(librosa.stft(y, n_fft=2048, hop_length=512))
    Sf = np.minimum(S, librosa.decompose.nn_filter(S, aggregate=np.median, metric='cosine',
                    width=int(librosa.time_to_frames(2, sr=sr, hop_length=512))))
    mask = librosa.util.softmask(S - Sf, 4 * Sf, power=2)
    fg = mask * S
    f = librosa.fft_frequencies(sr=sr, n_fft=2048)
    band = fg[(f > 200) & (f < 4000)]
    env = librosa.onset.onset_strength(S=librosa.amplitude_to_db(band, ref=np.max), sr=sr, hop_length=512)
    t = librosa.times_like(env, sr=sr, hop_length=512)
    pk = librosa.util.peak_pick(env, pre_max=3, post_max=3, pre_avg=5, post_avg=5, delta=0.3, wait=3)
    energy = band.sum(0); energy = energy / (energy.max() + 1e-9)
    return t[pk], env[pk] / (env.max() + 1e-9), t, energy

def align_line(words, t0, t1, on_t, on_s, lam=2.5):
    """DP: choose one onset per word, increasing, near a syllable-proportional prior."""
    syl = np.array([syllables(w) for w in words], float)
    span = min(t1 - t0, 0.38 * syl.sum() + 0.4)          # sung span estimate (~0.38 s / syllable)
    prior = t0 + span * np.r_[0, np.cumsum(syl)[:-1]] / syl.sum()
    m = (on_t >= t0 - 0.25) & (on_t < t1 - 0.05)
    ct, cs = on_t[m], on_s[m]
    n, k = len(words), len(ct)
    if k < n:  # not enough onsets: fall back to prior
        return prior.tolist()
    INF = 1e9; D = np.full((n, k), INF); B = np.zeros((n, k), int)
    cost = lambda i, j: -cs[j] + lam * abs(ct[j] - prior[i]) / max(span, 0.5)
    for j in range(k): D[0, j] = cost(0, j) + 3 * abs(ct[j] - t0)  # first word hugs LRC time
    for i in range(1, n):
        best, bj = INF, -1
        for j in range(k):
            if j >= 1 and D[i - 1, j - 1] < best: best, bj = D[i - 1, j - 1], j - 1
            if bj >= 0 and ct[j] - ct[bj] >= 0.09:
                D[i, j] = best + cost(i, j); B[i, j] = bj
    j = int(np.argmin(D[n - 1])); out = [0.0] * n
    for i in range(n - 1, -1, -1):
        out[i] = float(ct[j]); j = B[i, j]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('lrc'); ap.add_argument('audio')
    ap.add_argument('--start', type=float, default=0.0)
    ap.add_argument('--end', type=float, default=1e9)
    ap.add_argument('--out', default='words.json')
    a = ap.parse_args()
    raw = open(a.lrc, encoding='utf-8-sig').read()
    y, sr = librosa.load(a.audio, sr=22050, mono=True, offset=a.start,
                         duration=None if a.end > 1e8 else a.end - a.start + 3)
    dur = len(y) / sr
    on_t, on_s, et, en = vocal_onsets(y, sr)
    lines = [(t - a.start, s) for t, s in parse_lrc(a.lrc)]
    out = []
    for i, (t0, text) in enumerate(lines):
        t1 = lines[i + 1][0] if i + 1 < len(lines) else dur
        if not text or t0 >= a.end - a.start or t1 <= 0: continue
        enh = re.findall(r'<(\d+):(\d+(?:\.\d+)?)>\s*([^<]+)', text)
        if enh:
            words = [w.strip() for _, _, w in enh]; times = [int(m) * 60 + float(s) - a.start for m, s, _ in enh]
        else:
            words = text.split(); times = align_line(words, t0, t1, on_t, on_s)
        out.append({'line': i, 'start': round(t0, 3), 'end': round(t1, 3), 'text': ' '.join(words),
                    'words': [{'w': w, 't': round(t, 3)} for w, t in zip(words, times)]})
    beats_tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units='time')
    json.dump({'clip_start': a.start, 'clip_end': min(a.end, a.start + dur), 'tempo': float(np.atleast_1d(beats_tempo)[0]),
               'beats': [round(float(b), 3) for b in beats], 'lines': out}, open(a.out, 'w'), indent=1)
    for L in out:
        print(f"[{L['start']:6.2f}-{L['end']:6.2f}] " + '  '.join(f"{w['w']}@{w['t']:.2f}" for w in L['words']))

if __name__ == '__main__':
    main()
