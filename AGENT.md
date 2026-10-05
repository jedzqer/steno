# AGENT.md — AI Agent 项目速览

> 本文件面向 AI agent / 新接手的开发者，用于快速了解项目定位、结构和约定。

## 项目定位

**steno** 是一个 Windows 本地视频/音频转文字（速记）CLI 工具：

1. 接受本地媒体文件 **或视频网站 URL**（URL 时用 yt-dlp 自动下载到 `videos/`）
2. 用 FFmpeg 提取 16kHz 单声道音频
3. 用 SenseVoiceSmall 模型分块转录为文字，保存为 `_out.txt`

单文件核心脚本：`transcribe_video.py`。项目文档（README、本文件）均为中文。

## 目录结构

```text
steno/
├── transcribe_video.py      # 核心脚本：CLI 入口 + 全部逻辑（下载/提取/转录）
├── README.md                # 面向使用者的说明
├── AGENT.md                 # 本文件
├── .gitignore
├── SenseVoice-official/     # SenseVoice 官方仓库的本地 checkout（独立 git 仓库，已 gitignore，不入本仓库）
│   └── model.py             #   脚本通过 sys.path 指向此目录导入 SenseVoiceSmall
└── videos/                  # 媒体文件目录（已 gitignore）：示例视频、URL 下载目标、*_out.txt 输出
```

## 运行方式

```bash
python transcribe_video.py <本地文件路径 | 视频URL>
python transcribe_video.py <URL> --audio-only                  # 只下载音轨，更快
python transcribe_video.py <URL> --cookies-from-browser chrome # 需登录的站点
python transcribe_video.py <文件> -o out.txt --language zh     # 其余参数见 --help
```

## 环境依赖

- Windows + Python 3.13（代码兼容 3.8+）
- **FFmpeg**：必须在 PATH 中（提取音频）
- pip 包：`torch`、`soundfile`、`funasr`、`yt-dlp`（URL 下载）
- 模型权重：`MODEL_DIR` 常量指向 `C:\Users\jed\.cache\modelscope\hub\models\iic\SenseVoiceSmall`，运行前必须已存在
- GPU：自动检测 CUDA（`--device cpu` 可强制 CPU）

## transcribe_video.py 内部流程

```text
main()
 ├─ is_url(input)? ──是──> download_video()  # yt-dlp Python API，存到 videos/，已存在则跳过
 ├─ 按扩展名判断视频/音频（VIDEO_EXTENSIONS / AUDIO_EXTENSIONS）
 ├─ load_model()                             # SenseVoiceSmall.from_pretrained
 ├─ 视频 → ffmpeg 提取 16kHz wav 临时文件（转录后清理）
 └─ transcribe_audio()                       # 分块（默认 120s/块、1s 重叠）推理 → 拼接 → 过滤 emoji
```

## 关键设计决策（勿随意更改）

- **120 秒分块 + 1 秒重叠**：SenseVoiceSmall 的 CTC decoder 在约 120s 内表现最佳；更短会切断语句，更长对齐质量下降。`--chunk`/`--overlap` 可调
- **sys.path 注入**：`from model import SenseVoiceSmall` 依赖脚本把项目内 `SenseVoice-official/` 插入 sys.path；该目录是独立 git 仓库
- **yt-dlp 用 Python API 而非子进程**：`YoutubeDL(opts)` + `extract_info(url, download=True)`，最终路径取 `info['requested_downloads'][0]['filepath']`（含合并/后处理）。`noplaylist=True` 防止播放列表整包下载
- **emoji 过滤**：funasr 会把 SenseVoice 的情绪/事件标记（`<|ANGRY|>`、`<|BGM|>`）转成 emoji（😡🎼等），按 Unicode 区段 `U+1F000–U+1FFFF` 整体过滤，勿改回单字符枚举
- **stdout 编码兜底**：main() 入口对 stdout/stderr 做 `reconfigure(errors='replace')`，避免重定向/管道输出时 GBK 无法编码 emoji 而崩溃

## 注意事项 / 已知坑

- `videos/`、所有媒体文件和 `*_out.txt` 均被 gitignore，仓库里看不到测试产物是正常的
- Windows 控制台（GBK 代码页）下直接 `print` emoji 会崩，已有 `errors='replace'` 兜底；测试时管道捕获的中文乱码是显示问题，不是 bug
- yt-dlp 需保持更新（`pip install -U yt-dlp`）：站点解析规则随平台反爬变化频繁，旧版本常见 DownloadError
- 硬编码的绝对路径有两处：`MODEL_DIR`（模型缓存）和 SenseVoice sys.path 已改为相对脚本定位；若迁移环境需检查 `MODEL_DIR`
- URL 下载失败常见原因：站点需要登录（用 `--cookies-from-browser`）、yt-dlp 版本过旧、网络不可达
