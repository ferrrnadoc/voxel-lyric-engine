# voxel-lyric-engine

`voxel-lyric-oneshot` skill 的渲染引擎：用 CPU（numpy + PIL + librosa + ffmpeg）把歌曲音频和 LRC 歌词做成方块风、一镜到底的歌词 MV。方法论（画什么、怎么推导设定、关卡流程）在 skill 的 SKILL.md 里，这里只有代码。

## 用法

```bash
git clone --depth 1 https://github.com/ferrrnadoc/voxel-lyric-engine mv && cd mv
bash setup.sh                                   # 依赖、ffmpeg 检查、字体下载到 fonts/
python3 engine/lrc_words.py song.lrc song.mp3 --end 28 --out words.json
cp engine/story_template.py story.py
python3 story.py --camcheck                     # 运动上限和阅读诊断，不渲染
python3 story.py --audio song.mp3 --out out.mp4
```

## 文件

| 文件 | 作用 |
|---|---|
| `engine/lrc_words.py` | LRC + 音频 → 每个词被唱到的时间（words.json） |
| `engine/segment_scan.py` | 整首歌的行表、HOOK 标记和候选段落排名 |
| `engine/pitchseg.py` | 人声音符分段，编曲很满时用来手工对词 |
| `engine/vox.py` | 基础体素渲染器、关键帧镜头、天空 |
| `engine/vox2.py` | 扩展渲染器：move 动画、刚体组、旋转方块、运动模糊、字体栅格化 |
| `engine/kit.py` | 场景件：地面、树、人物、招牌、图标 |
| `engine/layout.py` | 歌词排布（弧、螺旋、球面、曲线、短语行）、按音节打字、粒子 |
| `engine/rail.py` | rail 镜头、从镜头反推字的位置、readcheck / stallcheck / steadycheck |
| `engine/camcheck.py` | 画面流速测量和 VIOLENT 上限 |
| `engine/keysheet.py` | 关键帧联系表 |
| `engine/story_template.py` | 能跑通全流程的基线 story，不是要交付的风格 |

音频、歌词、字体和渲染产物都不进仓库（见 `.gitignore`）。
