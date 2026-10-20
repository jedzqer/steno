# AGENT.md — AI Agent 项目速览

> 本文件面向 AI agent / 新接手的开发者，用于快速了解项目定位、结构和约定。

## 项目定位

**steno** 是一个 Windows 本地视频/音频转文字（速记）工具，提供 CLI 与 WebUI 两种入口：

1. 接受本地媒体文件 **或视频网站 URL**（URL 时用 yt-dlp 自动下载到 `videos/`）
2. 用 FFmpeg 提取 16kHz 单声道音频
3. 用 SenseVoiceSmall 模型分块转录为文字，保存为 `_out.txt`

核心脚本：`transcribe_video.py`（全部业务逻辑）；`webui.py` 是其上的 FastAPI 网页封装。项目文档（README、本文件）均为中文。

## 目录结构

```text
steno/
├── transcribe_video.py      # 核心脚本：CLI 入口 + 全部逻辑（下载/提取/转录）
├── webui.py                 # FastAPI 网页服务：任务队列/进度/历史 API，复用核心脚本
├── static/                  # WebUI 前端（无构建步骤的原生 HTML/CSS/JS）
│   ├── index.html
│   ├── style.css
│   └── app.js
├── smoke_test.py            # 冒烟测试：降级链 / 重试接口 / 媒体复用 / 任务上限（python smoke_test.py）
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

python webui.py                       # WebUI：默认 http://127.0.0.1:8321，自动开浏览器
python webui.py --port 9000 --no-browser
```

## 环境依赖

- Windows + Python 3.13（代码兼容 3.8+）
- **FFmpeg**：必须在 PATH 中（提取音频、解码 m4a/aac 等 libsndfile 不支持的格式）
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
- **重型依赖惰性导入**：`torch`（单独 import 约 10s）、`funasr`、`SenseVoice-official/model.py`、`soundfile`、`numpy` 一律在用到的函数内部 import。模块顶层只保留轻量 import，因此 `python transcribe_video.py --help` 与 `import transcribe_video` 均为 0.1s 级，而 WebUI 启动不再被模型依赖拖到 50s+
- **模型后台预加载**：WebUI 启动后用后台线程加载模型（`--no-preload` 可关闭），`/api/info` 里的 `model_loading` / `cuda_checked` 反映探测进度；`_MODELS` 仍按设备缓存，同一设备只加载一次
- **sys.path 注入**：`from model import SenseVoiceSmall` 依赖脚本把项目内 `SenseVoice-official/` 插入 sys.path；该目录是独立 git 仓库
- **yt-dlp 用 Python API 而非子进程**：`YoutubeDL(opts)` + `extract_info(url, download=True)`，最终路径取 `info['requested_downloads'][0]['filepath']`（含合并/后处理）。`noplaylist=True` 防止播放列表整包下载。`download_video()` 另有可选 `progress_cb(status, downloaded, total, speed)` 供 WebUI 展示进度（默认 None，不影响 CLI）
- **WebUI 架构**：`webui.py` 完全复用 `transcribe_video.py`（导入为模块），不重复实现业务逻辑。单 worker 线程串行消费任务队列（GPU 推理串行化）；模型进程内缓存只加载一次；取消通过 `stop_event`（转录阶段核心脚本原生支持；下载阶段 WebUI 在 progress_cb 里抛 `yt_dlp.utils.DownloadCancelled` 沿 hook 传播中断）；前端为 static/ 下原生三件套，无构建步骤，轮询 `/api/jobs` 展示进度
- **emoji 过滤**：funasr 会把 SenseVoice 的情绪/事件标记（`<|ANGRY|>`、`<|BGM|>`）转成 emoji（😡🎼等），按 Unicode 区段 `U+1F000–U+1FFFF` 整体过滤，勿改回单字符枚举
- **stdout 编码兜底**：main() 入口对 stdout/stderr 做 `reconfigure(errors='replace')`，避免重定向/管道输出时 GBK 无法编码 emoji 而崩溃

## 转录降级链（WebUI）

转录阶段失败不会直接放弃，而是按固定阶梯逐级降级重试，由 `webui.py` 的 `_next_attempt()` 计算下一次尝试的参数：

```text
OOM / 显存不足          → 立刻回落 CPU（其余参数不变）
其他异常（首次）        → 同参数重试一次
仍然异常                → 关闭 ITN（use_itn=False）后重试
仍然异常                → 分块减半（120 → 60 → 30，下限 30s）后重试
仍然异常                → 回落 CPU 重试
```

- 每次降级都会更新 `job.message`（前端可见），并调用 `torch.cuda.empty_cache()` 释放显存
- 空结果（无异常但转出空文本）不浪费一次同参数重试，直接走“关闭 ITN”
- 阶梯耗尽后抛 `RuntimeError(f"转录失败（已尝试 {n} 次）：...")`，任务标记为 error，前端显示错误摘要 + “重试”按钮

## 重试 / 复用媒体

- `Job.media_path` 记录实际使用的本地媒体路径（上传或下载所得），`to_dict()` 额外返回 `has_media` / `media`
- `POST /api/jobs/{id}/retry`：媒体文件仍在 → 以 `kind="local"` 新建任务，**跳过下载阶段**（重新转录不重新下载）；媒体已丢失且原任务是 URL → 重新下载（`kind="url"`）；其它情况 → HTTP 400
- `POST /api/history/retry?name=<xxx_out.txt>`：从 `videos/` 里按文件名主干找源媒体（兼容 `xxx (2)_out.txt` 这种重名后缀），复用它重新转录
- 前端：任务列表错误行显示错误摘要 + “重试”，已完成且有媒体的任务显示“重新转录”，历史卡片显示“重新转录”
- `download_video(..., skip_existing=True)`：先 `extract_info(download=False)` 取文件名，若文件已存在且非空且没有残留 `.part`，直接复用而不下载

## 稳定性细节

- `JOBS` 为内存字典，上限 `MAX_JOBS = 200`，`_prune_jobs()` 在入队时淘汰最旧的已结束任务，避免长期运行内存无限增长
- 下载失败已有重试（`DOWNLOAD_ATTEMPTS = 3`、`DOWNLOAD_RETRY_DELAY = 4`，yt-dlp 自身还有 `retries=10` 等）
- 下载阶段的取消检查放在成功返回之前，**不放在 finally 里**（finally 抛异常会覆盖成功结果）

## 注意事项 / 已知坑

- libsndfile（soundfile 底层）不支持 AAC/M4A/WMA 等格式，`sf.read` 会报 `Format not recognised`；
  且 SenseVoice 要求 16kHz 输入。因此 `transcribe_audio()` 对 soundfile 读不出或非 16kHz 的音频
  统一回落到 `decode_audio_with_ffmpeg()`（ffmpeg 管道解码 + 重采样到 16kHz），勿绕过；
  否则 yt-dlp `--audio-only` 下载的 m4a（通常是 44.1kHz AAC）会在读取阶段直接失败

- `videos/`、所有媒体文件和 `*_out.txt` 均被 gitignore，仓库里看不到测试产物是正常的
- Windows 控制台（GBK 代码页）下直接 `print` emoji 会崩，已有 `errors='replace'` 兜底；测试时管道捕获的中文乱码是显示问题，不是 bug
- yt-dlp 需保持更新（`pip install -U yt-dlp`）：站点解析规则随平台反爬变化频繁，旧版本常见 DownloadError
- 硬编码的绝对路径有两处：`MODEL_DIR`（模型缓存）和 SenseVoice sys.path 已改为相对脚本定位；若迁移环境需检查 `MODEL_DIR`
- URL 下载失败常见原因：站点需要登录（用 `--cookies-from-browser`）、yt-dlp 版本过旧、网络不可达
