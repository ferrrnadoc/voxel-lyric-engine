#!/usr/bin/env python3
"""pitchseg.py - voiced-note segments of the lead vocal (pYIN on the REPET-SIM foreground), for hand-mapping words
when lrc_words.py's onset DP snaps to drums (dense mixes). Prints one row per sung note: start, end, length, pitch.
Usage: python3 pitchseg.py song.mp3 --start 110 --end 151 [--lrc song.lrc]   (times printed relative to --start)
Map each LRC word to a segment start by reading the lines in order; then CHECK BY EAR - the user's ear wins."""
import argparse, re, numpy as np, librosa

def foreground(y, sr):
    S, ph = librosa.magphase(librosa.stft(y, n_fft=2048, hop_length=512))
    Sf = np.minimum(S, librosa.decompose.nn_filter(S, aggregate=np.median, metric='cosine',
                    width=int(librosa.time_to_frames(2, sr=sr, hop_length=512))))
    mask = librosa.util.softmask(S - Sf, 4 * Sf, power=2)
    return librosa.istft(mask * S * ph, hop_length=512, length=len(y))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('audio'); ap.add_argument('--lrc')
    ap.add_argument('--start', type=float, default=0); ap.add_argument('--end', type=float, default=30)
    a = ap.parse_args()
    y, sr = librosa.load(a.audio, sr=22050, offset=a.start, duration=a.end - a.start)
    fg = foreground(y, sr); hop = 256
    f0, vf, vp = librosa.pyin(fg, fmin=75, fmax=900, sr=sr, frame_length=2048, hop_length=hop)
    t = librosa.times_like(f0, sr=sr, hop_length=hop)
    midi = np.where(vf, librosa.hz_to_midi(np.nan_to_num(f0, nan=1.0)), np.nan)
    segs, i = [], 0
    while i < len(t):                      # split at unvoiced gaps or pitch jumps > 1.5 semitones
        if np.isnan(midi[i]): i += 1; continue
        j = i + 1
        while j < len(t) and not np.isnan(midi[j]) and abs(midi[j] - np.nanmedian(midi[i:j])) < 1.5: j += 1
        if t[j - 1] - t[i] >= 0.06: segs.append((t[i], t[min(j, len(t) - 1)], float(np.nanmedian(midi[i:j]))))
        i = j
    if a.lrc:
        for raw in open(a.lrc, encoding='utf-8-sig'):
            for m, s in re.findall(r'\[(\d+):(\d+(?:\.\d+)?)\]', raw):
                tt = int(m) * 60 + float(s) - a.start
                txt = re.sub(r'\[[^\]]*\]', '', raw).strip()
                if 0 <= tt < a.end - a.start: print(f'# LRC {tt:6.2f}  {txt}')
    print(' start    end   len  note')
    for s, e, m in segs:
        print(f'{s:6.2f} {e:6.2f} {e - s:5.2f}  {librosa.midi_to_note(round(m))}')

if __name__ == '__main__':
    main()
