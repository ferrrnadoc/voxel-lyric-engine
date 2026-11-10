#!/usr/bin/env bash
# One-time setup for a working copy: Python deps, ffmpeg check, fonts into ./fonts (vox2 reads engine/../fonts).
set -u
cd "$(dirname "$0")"
pip install -q -r requirements.txt --break-system-packages 2>/dev/null || pip install -q -r requirements.txt
command -v ffmpeg >/dev/null || echo "!! ffmpeg not found - install it before rendering"
mkdir -p fonts && cd fonts
B=https://raw.githubusercontent.com/google/fonts/main   # raw.githubusercontent.com only; github.com/.../raw is blocked in some sandboxes
for p in ofl/silkscreen/Silkscreen-Bold.ttf ofl/pixelifysans/PixelifySans%5Bwght%5D.ttf ofl/vt323/VT323-Regular.ttf \
  ofl/jersey15/Jersey15-Regular.ttf ofl/fredoka/Fredoka%5Bwdth,wght%5D.ttf ofl/nunito/Nunito%5Bwght%5D.ttf \
  ofl/bungee/Bungee-Regular.ttf apache/robotoslab/RobotoSlab%5Bwght%5D.ttf ofl/cinzel/Cinzel%5Bwght%5D.ttf \
  ofl/jacquardabastarda9/JacquardaBastarda9-Regular.ttf ofl/bebasneue/BebasNeue-Regular.ttf \
  ofl/caveatbrush/CaveatBrush-Regular.ttf ofl/archivoblack/ArchivoBlack-Regular.ttf ofl/oswald/Oswald%5Bwght%5D.ttf \
  ofl/rye/Rye-Regular.ttf ofl/rubik/Rubik%5Bwght%5D.ttf ofl/overpass/Overpass%5Bwght%5D.ttf \
  ofl/courierprime/CourierPrime-Bold.ttf; do
  f=$(basename "$p" | sed 's/%5B/[/g; s/%5D/]/g')
  [ -s "$f" ] || curl -sfL "$B/$p" -o "$f" || echo "missing $f"
done
echo "setup done: $(ls | wc -l) fonts in ./fonts"
