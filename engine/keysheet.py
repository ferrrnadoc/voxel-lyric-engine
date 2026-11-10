#!/usr/bin/env python3
"""keysheet.py - gate-2 keyframe contact sheet: one row per lyric line, 2-4 preview frames, a caption under each
frame saying HOW the lyric is displayed there (carrier / type / entry action).
Usage: python3 keysheet.py sheet.json   with sheet.json =
  {"title": "...", "subtitle": "...", "frames": "out_prev_{fi:05d}.png", "fps": 30, "out": "keyframes.jpg",
   "rows": [["1:50", "lyric line text", [[1.3, "caption"], [3.0, "caption"]]], ...]}
Render the frames first: python3 story.py --preview 1.3,3.0,... --size 540 (file index = round(t*30))"""
import json, os, sys
from PIL import Image, ImageDraw, ImageFont

def F(sz, bold=False):
    for p in ('/usr/share/fonts/opentype/noto/NotoSansCJK-%s.ttc' % ('Bold' if bold else 'Regular'),
              '/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf' % ('-Bold' if bold else '')):
        if os.path.exists(p): return ImageFont.truetype(p, sz, index=2 if p.endswith('.ttc') else 0)
    return ImageFont.load_default()

def wrap(d, text, font, width):
    """character-level wrap (works for CJK); prefers breaking at the last space"""
    out = []
    for para in text.split('\n'):
        line = ''
        for ch in para:
            if d.textlength(line + ch, font=font) <= width: line += ch; continue
            k = line.rfind(' ')
            if k > 0: out.append(line[:k]); line = line[k + 1:] + ch
            else: out.append(line); line = ch
        out.append(line)
    return '\n'.join(out)

def main(spec):
    S = json.load(open(spec)); base = os.path.dirname(os.path.abspath(spec))
    TH, CAPW, GAP, CAPH = 430, 400, 14, 50
    ncol = max(len(r[2]) for r in S['rows'])
    W_ = CAPW + ncol * (TH + GAP) + GAP; H_ = 120 + len(S['rows']) * (TH + CAPH + GAP + 6) + 20
    im = Image.new('RGB', (W_, H_), (24, 24, 30)); d = ImageDraw.Draw(im)
    d.text((GAP, 22), S.get('title', ''), font=F(40, True), fill=(245, 236, 220))
    d.text((GAP, 76), S.get('subtitle', ''), font=F(24), fill=(170, 170, 180))
    y = 120
    for tc, lyr, frames in S['rows']:
        d.text((GAP, y + 8), tc, font=F(30, True), fill=(255, 196, 120))
        d.multiline_text((GAP, y + 54), wrap(d, lyr, F(21), CAPW - 2 * GAP), font=F(21), fill=(235, 235, 240), spacing=8)
        for c, (t, cap) in enumerate(frames):
            fi = int(round(t * S.get('fps', 30)))
            fr = Image.open(os.path.join(base, S['frames'].format(fi=fi))).convert('RGB').resize((TH, TH), Image.LANCZOS)
            x = CAPW + c * (TH + GAP); im.paste(fr, (x, y))
            d.rectangle([x, y, x + 64, y + 26], fill=(0, 0, 0)); d.text((x + 6, y + 1), f'{t:.1f}s', font=F(18), fill=(255, 230, 120))
            d.multiline_text((x, y + TH + 6), wrap(d, cap, F(20), TH), font=F(20), fill=(220, 220, 228), spacing=2)
        y += TH + CAPH + GAP + 6
    out = os.path.join(base, S.get('out', 'keyframes.jpg')); im.save(out, quality=88); print(out, im.size)

if __name__ == '__main__':
    main(sys.argv[1])
