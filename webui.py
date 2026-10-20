"""steno WebUI — 基于 FastAPI 的本地网页界面。

启动: python webui.py [--host 127.0.0.1] [--port 8321] [--no-browser]

功能:
  - 上传本地媒体文件，或提交视频 URL（yt-dlp 自动下载）
  - 后台单 worker 队列依次转录，实时展示 下载/提取音频/转录 进度
  - 任务取消、失败原因展示
  - 历史记录（videos/*_out.txt）浏览 / 查看 / 删除
  - 模型进程内缓存（按推理设备各缓存一份，切换 GPU/CPU 真实生效）

完全复用 transcribe_video.py 的下载 / 提取 / 转录逻辑，不重复实现。
"""
import argparse
import importlib.util
import os
import queue
import re
import shutil
import sys
import tempfile
import threading
import time
import traceback
import uuid
import webbrowser
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DOWNLOAD_DIR = BASE_DIR / "videos"

# ---------------------------------------------------------------------------
# 引入核心脚本。torch/funasr 缺失时界面仍可启动，仅转录功能给出明确报错。
# ---------------------------------------------------------------------------
TV = None
TV_ERROR = None
try:
    sys.path.insert(0, str(BASE_DIR))
    import transcribe_video as tv  # noqa: E402  (依赖脚本自身的 sys.path 注入)
    TV = tv
    DOWNLOAD_DIR = tv.DOWNLOAD_DIR
except Exception as e:  # pragma: no cover - 仅在依赖损坏时触发
    TV_ERROR = f"{type(e).__name__}: {e}"

app = FastAPI(title="steno WebUI", docs_url=None, redoc_url=None)


# ---------------------------------------------------------------------------
# 任务模型与后台队列（单 worker，GPU 推理串行化）
# ---------------------------------------------------------------------------
class _Cancelled(Exception):
    """任务被用户取消（内部控制流）。"""


class _Attempt(TypedDict):
    """一次转录尝试的参数（降级链的节点）。"""
    device: str
    use_itn: bool
    chunk: int
    overlap: int
    why: str
    retried: bool


class Job:
    id: str
    kind: str
    source: str
    options: dict
    status: str
    stage: str
    message: str
    progress: float | None
    downloaded: int | None
    total: int | None
    speed: float | None
    result_name: str | None
    error: str | None
    media_path: str | None
    created: float
    started: float | None
    finished: float | None

    def __init__(self, kind: str, source: str, options: dict):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind                  # 'upload' | 'url' | 'local'
        self.source = source              # 上传/本地文件的绝对路径 / 视频 URL
        self.options = options
        self.status = "queued"            # queued/running/done/error/cancelled
        self.stage = ""                   # download/load/extract/transcribe
        self.message = "排队中…"
        self.progress = None              # 0~1；None 表示不定进度
        self.downloaded = None            # 下载字节
        self.total = None
        self.speed = None                 # B/s
        self.result_name = None           # 结果 txt 文件名（不含路径）
        self.error = None
        # 媒体文件的落盘路径：下载或上传完成后写入。重试时据此跳过下载阶段，
        # 直接复用已下载的媒体（重新转录不重新下载）。
        self.media_path = None
        self.created = time.time()
        self.started = None
        self.finished = None
        self.stop_event = threading.Event()

    def display_name(self) -> str:
        if self.kind == "url":
            text = self.source
            for prefix in ("https://", "http://", "www."):
                if text.startswith(prefix):
                    text = text[len(prefix):]
            return text if len(text) <= 64 else text[:61] + "…"
        return Path(self.source).name

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "display": self.display_name(),
            "status": self.status,
            "stage": self.stage,
            "message": self.message,
            "progress": self.progress,
            "downloaded": self.downloaded,
            "total": self.total,
            "speed": self.speed,
            "result": self.result_name,
            "error": self.error,
            "has_media": bool(self.media_path) and Path(self.media_path).exists(),
            "media": Path(self.media_path).name if self.media_path else None,
            "created": self.created,
            "started": self.started,
            "finished": self.finished,
        }


JOBS: dict[str, Job] = {}
JOB_QUEUE: "queue.Queue[str]" = queue.Queue()
_MODELS: dict[str, tuple] = {}      # 解析后的设备名 -> (model, kwargs)
_MODEL_LOCK = threading.Lock()


def _check_cancel(job: Job):
    if job.stop_event.is_set():
        raise _Cancelled()


def _ensure_model(device: str):
    """按推理设备缓存模型：同一设备只加载一次；切换 GPU/CPU 会真实生效。"""
    if TV is None:
        raise RuntimeError(f"后端依赖不可用，无法转录：{TV_ERROR}")
    resolved = TV.resolve_device(device)
    with _MODEL_LOCK:
        if resolved not in _MODELS:
            _MODELS[resolved] = TV.load_model(device=resolved)
        return _MODELS[resolved]


_MODEL_LOADING = False      # 后台预加载进行中（供 /api/info 展示）


def _preload_model_async(device: str = "auto"):
    """后台线程预加载模型：把 torch 导入 + SenseVoice 权重加载的一次性成本
    挪到启动之后的后台，不阻塞 WebUI 打开；第一个任务进来时通常已就绪。"""
    global _MODEL_LOADING
    if TV is None:
        return
    _MODEL_LOADING = True
    try:
        print("[preload] 正在后台加载模型（torch 导入 + SenseVoice 权重）...")
        _ensure_model(device)
        print("[preload] 模型预加载完成")
    except Exception as e:
        print(f"[preload] 模型预加载失败（不影响使用，转录时会重试）: {e}")
    finally:
        _MODEL_LOADING = False


@lru_cache(maxsize=1)
def _cuda_available() -> bool:
    """CUDA 是否可用（结果缓存，torch 的 import 约 10s，不该每次 /api/info 都付这个代价）。"""
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


_CUDA_RESULT: list[bool | None] = [None]     # None = 尚未探测完成


def _warm_cuda_async():
    """后台探测 CUDA 可用性：import torch 较慢，不能在 HTTP 请求里同步等。"""
    _CUDA_RESULT[0] = _cuda_available()


def _cuda_state() -> bool | None:
    return _CUDA_RESULT[0]


DOWNLOAD_ATTEMPTS = 3        # 下载总尝试次数（首次 + 2 次整体重试）
DOWNLOAD_RETRY_DELAY = 4     # 重试前的等待秒数
MAX_JOBS = 200               # 内存中保留的任务上限（超出后淘汰最旧的已结束任务）


def _sleep_cancellable(seconds: float, stop_event: threading.Event) -> None:
    """分段睡眠等待，期间被取消立即抛 _Cancelled。"""
    deadline = time.monotonic() + seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        if stop_event.is_set():
            raise _Cancelled()
        time.sleep(min(0.2, remaining))


def _download(job: Job) -> Path:
    if TV is None:
        raise RuntimeError(f"后端依赖不可用，无法转录：{TV_ERROR}")
    o = job.options

    def cb(status, downloaded, total, speed):
        # 取消：抛 DownloadCancelled 会沿 yt-dlp 的 progress hook 传播，干净地中断下载
        if job.stop_event.is_set():
            from yt_dlp.utils import DownloadCancelled
            raise DownloadCancelled("用户取消")
        if status == "downloading":
            job.downloaded, job.total, job.speed = downloaded, total, speed
            job.progress = (downloaded / total) if total else None

    for attempt in range(DOWNLOAD_ATTEMPTS):
        if job.stop_event.is_set():
            raise _Cancelled()
        try:
            path = TV.download_video(
                job.source,
                audio_only=o.get("audio_only", True),
                cookies_from_browser=o.get("cookies") or None,
                progress_cb=cb,
            )
        except Exception as e:
            # 只对 yt-dlp 的下载失败（download_video 包装成 RuntimeError）做整体重试，
            # 其余异常（含取消）原样抛出。每次重试都会重新请求 playurl，
            # 常会分配到比上次更健康的 CDN 节点，对 mcdn/PCDN 抖动特别有效。
            if attempt == DOWNLOAD_ATTEMPTS - 1 or not isinstance(e, RuntimeError):
                raise
            if job.stop_event.is_set():
                raise _Cancelled() from None
            job.message = (f"下载出错，{DOWNLOAD_RETRY_DELAY} 秒后重试"
                           f"（{attempt + 1}/{DOWNLOAD_ATTEMPTS - 1}）…")
            print(f"[download] 第 {attempt + 1}/{DOWNLOAD_ATTEMPTS} 次下载失败，"
                  f"{DOWNLOAD_RETRY_DELAY} 秒后重试: {e}")
            _sleep_cancellable(DOWNLOAD_RETRY_DELAY, job.stop_event)
            job.message = "下载中…"
            continue
        # 成功：先检查取消再返回（早前用 finally + raise 覆盖返回值的写法是隐雷）
        if job.stop_event.is_set():
            raise _Cancelled()
        return path
    raise RuntimeError("下载失败：重试次数已用完")   # 防御：循环内必 return 或 raise


def _is_oom(exc: BaseException) -> bool:
    """判断异常是否为显存不足（CUDA OOM）——这类错误可以回落 CPU 重试。"""
    msg = f"{type(exc).__name__}: {exc}".lower()
    return ("out of memory" in msg or "cuda error" in msg or
            "cuda oom" in msg or "cudnn_status_alloc_failed" in msg or
            "cublas_status_alloc_failed" in msg)


def _free_cuda_cache():
    """OOM 后主动释放显存，避免重试时因碎片化再次失败。"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:       # 清理失败不应影响主流程
        print(f"[cuda] 释放显存失败（已忽略）: {e}")


def _next_attempt(prev: _Attempt, exc: BaseException | None, empty: bool) -> _Attempt | None:
    """根据上一次失败的原因，给出下一次尝试的参数（降级链）。

    降级顺序：OOM → 回落 CPU；其它异常 → 同参数重试一次；仍失败 → 关闭 ITN；
    再失败 → 分块减半；再失败 → CPU + 小分块。空结果（无异常）时直接从
    “关闭 ITN”开始（重试相同参数没意义）。
    """
    retried = prev.get("retried", False)
    oom = exc is not None and _is_oom(exc)
    if oom and prev["device"] != "cpu":
        return _Attempt(device="cpu", use_itn=prev["use_itn"], chunk=prev["chunk"],
                        overlap=prev["overlap"], retried=retried,
                        why="显存不足，回落 CPU 重试")
    if not empty and not retried:
        return _Attempt(device=prev["device"], use_itn=prev["use_itn"], chunk=prev["chunk"],
                        overlap=prev["overlap"], retried=True, why="同一参数重试一次")
    if prev["use_itn"]:
        return _Attempt(device=prev["device"], use_itn=False, chunk=prev["chunk"],
                        overlap=prev["overlap"], retried=retried, why="关闭 ITN 后重试")
    if prev["chunk"] > 30:
        return _Attempt(device=prev["device"], use_itn=prev["use_itn"],
                        chunk=max(30, prev["chunk"] // 2), overlap=prev["overlap"],
                        retried=retried, why="分块减半后重试")
    if prev["device"] != "cpu":
        return _Attempt(device="cpu", use_itn=prev["use_itn"], chunk=prev["chunk"],
                        overlap=prev["overlap"], retried=retried, why="回落 CPU 重试")
    return None


def _as_int(value, default: int) -> int:
    """把选项里的数值安全地转成 int（选项可能来自表单/历史记录，类型不保证）。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _transcribe_with_fallbacks(job: Job, audio_path: str) -> str:
    """带降级链的转录：失败时自动换参数/换设备重试，而不是直接判任务失败。

    每次尝试都更新 job.message，前端可见当前处于第几次尝试、为何降级。
    """
    tv = TV
    if tv is None:
        raise RuntimeError(f"后端依赖不可用，无法转录：{TV_ERROR}")
    o = job.options
    attempt: _Attempt | None = _Attempt(
        device=str(o.get("device", "auto")),
        use_itn=bool(o.get("use_itn", True)),
        chunk=_as_int(o.get("chunk", 120), 120),
        overlap=_as_int(o.get("overlap", 1), 1),
        why="首次尝试",
        retried=False,
    )
    last_err: BaseException | None = None
    n = 0
    while attempt is not None:
        n += 1
        if n > 1:
            job.message = f"转录失败，{attempt['why']}（第 {n} 次尝试）…"
        else:
            job.message = "转录中…"
        job.progress = 0.0
        empty = False
        try:
            resolved = tv.resolve_device(attempt["device"])
            model, kwargs = _ensure_model(resolved)
            _check_cancel(job)

            def on_progress(i, total, name, _job=job, _why=attempt["why"], _n=n):
                _check_cancel(_job)
                _job.progress = i / total if total else None
                _job.message = (f"转录中 · 第 {i}/{total} 段"
                                + (f"（{_why}）" if _n > 1 else ""))

            text = tv.transcribe_audio(
                audio_path, model, kwargs,
                language=str(o.get("language", "zh")),
                use_itn=attempt["use_itn"],
                ban_emo_unk=False,
                chunk_sec=attempt["chunk"],
                overlap_sec=attempt["overlap"],
                progress_cb=on_progress,
                stop_event=job.stop_event,
            )
            if text and text.strip():
                return text
            # 空结果：当作一种可恢复失败，继续降级重试
            last_err = RuntimeError("转录结果为空（音频可能没有可识别的语音）")
            empty = True
        except _Cancelled:
            raise
        except tv.TranscriptionCancelled:
            raise _Cancelled() from None
        except Exception as e:
            last_err = e
            empty = False
            print(f"[transcribe] 第 {n} 次尝试失败: {type(e).__name__}: {e}")
            if _is_oom(e):
                _free_cuda_cache()
        attempt = _next_attempt(attempt, last_err, empty)
    raise RuntimeError(f"转录失败（已尝试 {n} 次）：{type(last_err).__name__}: {last_err}")


def _run_job(job: Job):
    if TV is None:
        raise RuntimeError(f"后端依赖不可用，无法转录：{TV_ERROR}")

    job.status = "running"
    job.started = time.time()
    job.message = "准备中…"

    if job.kind == "url":
        job.stage, job.message = "download", "下载中…"
        job.progress = None
        input_path = _download(job)
    else:
        # 'upload'：上传后已落盘；'local'：复用既有媒体（重试/重新转录）
        input_path = Path(job.source)
        if not input_path.exists():
            raise RuntimeError(f"文件不存在: {input_path}")
    # 记录媒体落盘路径：重试时据此跳过下载阶段（重新转录不重新下载）
    job.media_path = str(input_path)
    _check_cancel(job)

    suffix = input_path.suffix.lower()
    if suffix not in TV.VIDEO_EXTENSIONS and suffix not in TV.AUDIO_EXTENSIONS:
        raise RuntimeError(f"不支持的文件格式 '{suffix or '(无后缀)'}'")

    _check_cancel(job)
    job.stage, job.message, job.progress = "load", "加载模型…", None
    _ensure_model(job.options.get("device", "auto"))   # 预热：让“加载模型”阶段可见
    _check_cancel(job)

    temp_audio = None
    if suffix in TV.VIDEO_EXTENSIONS:
        job.stage, job.message, job.progress = "extract", "提取音频…", None
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_audio = f.name
        try:
            # transcribe_video.extract_audio_from_video(..., stop_event=...) 会在取消时
            # kill ffmpeg 并抛出 TranscriptionCancelled，使本阶段取消即时生效。
            ok = TV.extract_audio_from_video(str(input_path), temp_audio,
                                             stop_event=job.stop_event)
        except TV.TranscriptionCancelled:    # 取消即时生效：ffmpeg 进程已被 kill
            _cleanup_temp(temp_audio)
            temp_audio = None
            raise _Cancelled() from None
        except BaseException:                 # 其它失败也要清掉半成品 wav
            _cleanup_temp(temp_audio)
            temp_audio = None
            raise
        if not ok:
            _cleanup_temp(temp_audio)
            temp_audio = None
            raise RuntimeError("ffmpeg 提取音频失败（请确认 ffmpeg 已加入 PATH）")
        audio_path = temp_audio
    else:
        audio_path = str(input_path)

    try:
        _check_cancel(job)
        job.stage, job.message, job.progress = "transcribe", "转录中…", 0.0
        text = _transcribe_with_fallbacks(job, audio_path)

        out_path = _unique_out_path(input_path)   # 重名不静默覆盖，追加 (2)/(3)…
        out_path.write_text(text, encoding="utf-8")
        job.result_name = out_path.name
        job.status, job.message, job.progress = "done", "完成", 1.0
    finally:
        if temp_audio:
            _cleanup_temp(temp_audio)


def _worker_loop():
    while True:
        job_id = JOB_QUEUE.get()
        job = JOBS.get(job_id)
        if job is None:
            continue
        if job.stop_event.is_set():          # 排队期间被取消
            job.status, job.message = "cancelled", "已取消"
            job.finished = time.time()
            continue
        try:
            _run_job(job)
        except _Cancelled:
            job.status, job.message = "cancelled", "已取消"
        except BaseException as e:           # 兜住 SystemExit（yt-dlp 缺失时核心脚本 sys.exit）
            if job.stop_event.is_set():
                job.status, job.message = "cancelled", "已取消"
            else:
                job.status = "error"
                job.message = "失败"
                job.error = f"{type(e).__name__}: {e}"
                traceback.print_exc()        # 失败详情同步打印到控制台，便于排查
        finally:
            if job.finished is None:
                job.finished = time.time()
            job.downloaded = job.total = job.speed = None


def _start_worker():
    threading.Thread(target=_worker_loop, name="steno-worker", daemon=True).start()

_start_worker()
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def _unique_dest(directory: Path, filename: str) -> Path:
    name = Path(filename).name or "upload.bin"
    candidate = directory / name
    if not candidate.exists():
        return candidate
    stem, suffix = candidate.stem, candidate.suffix
    for i in range(2, 1000):
        candidate = directory / f"{stem} ({i}){suffix}"
        if not candidate.exists():
            return candidate
    raise HTTPException(400, "文件名冲突过多")


def _cleanup_temp(path) -> None:
    try:
        os.unlink(path)
    except OSError as e:
        print(f"[cleanup] 临时文件删除失败（已忽略）: {e}")


def _unique_out_path(input_path: Path) -> Path:
    """转录结果输出路径；重名时追加 (2)/(3)…，不静默覆盖旧结果。"""
    base = input_path.with_name(f"{input_path.stem}_out.txt")
    if not base.exists():
        return base
    for i in range(2, 1000):
        cand = input_path.with_name(f"{input_path.stem} ({i})_out.txt")
        if not cand.exists():
            return cand
    return base   # 极端情况（重名上千次）退回覆盖


def _resolve_history_file(name: str) -> Path:
    if not name or not name.endswith("_out.txt"):
        raise HTTPException(400, "无效的文件名")
    base = DOWNLOAD_DIR.resolve()
    target = (base / name).resolve()
    if target.parent != base or not target.exists():
        raise HTTPException(404, "记录不存在")
    return target


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/info")
def info():
    yt_dlp_ok = importlib.util.find_spec("yt_dlp") is not None
    return {
        "backend_ok": TV is not None,
        "backend_error": TV_ERROR,
        "cuda": _cuda_state(),
        "cuda_checked": _cuda_state() is not None,
        "model_loaded": bool(_MODELS),
        "model_loading": _MODEL_LOADING,
        "model_devices": sorted(_MODELS),
        "model_dir_exists": TV is not None and Path(TV.MODEL_DIR).exists(),
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "yt_dlp": yt_dlp_ok,
        "download_dir": str(DOWNLOAD_DIR),
    }


DEFAULT_OPTIONS = {
    "audio_only": True,
    "cookies": "",
    "language": "zh",
    "device": "auto",
    "use_itn": True,
    "chunk": 120,
    "overlap": 1,
}


def _enqueue_job(kind: str, source: str, options: dict) -> Job:
    """登记并排队一个任务；同时清理超出上限的旧任务（内存不无限增长）。"""
    job = Job(kind, source, options)
    JOBS[job.id] = job
    JOB_QUEUE.put(job.id)
    _prune_jobs()
    return job


def _prune_jobs():
    """JOBS 是纯内存字典：长时间运行会无限增长，淘汰最旧的已结束任务。"""
    if len(JOBS) <= MAX_JOBS:
        return
    finished = [j for j in JOBS.values()
                if j.status in ("done", "error", "cancelled")]
    finished.sort(key=lambda j: j.created)
    for j in finished[:max(0, len(JOBS) - MAX_JOBS)]:
        JOBS.pop(j.id, None)


@app.post("/api/jobs")
async def create_job(
    mode: str = Form(...),                      # 'file' | 'url'
    url: str = Form(""),
    file: UploadFile | None = File(None),
    audio_only: bool = Form(True),
    cookies: str = Form(""),
    language: str = Form("zh"),
    device: str = Form("auto"),
    use_itn: bool = Form(True),
    chunk: int = Form(120),
    overlap: int = Form(1),
):
    if mode not in ("file", "url"):
        raise HTTPException(400, "mode 必须是 file 或 url")
    if mode == "url":
        url = url.strip()
        if not url:
            raise HTTPException(400, "请填写视频链接")
        if not (url.startswith(("http://", "https://", "www."))):
            raise HTTPException(400, "链接需以 http(s):// 或 www. 开头")
        if importlib.util.find_spec("yt_dlp") is None:
            raise HTTPException(400, "未安装 yt-dlp（pip install yt-dlp）")
        source = url
    else:
        if file is None or not file.filename:
            raise HTTPException(400, "请选择文件")
        dest = _unique_dest(DOWNLOAD_DIR, file.filename)
        try:
            with open(dest, "wb") as out:
                while chunk_bytes := await file.read(4 * 1024 * 1024):
                    out.write(chunk_bytes)
        except OSError as e:
            dest.unlink(missing_ok=True)   # 清理写了一半的残留文件
            raise HTTPException(500, f"保存上传文件失败: {e}") from e
        source = str(dest)

    job = _enqueue_job(mode, source, {
        "audio_only": audio_only,
        "cookies": cookies,
        "language": language,
        "device": device,
        "use_itn": use_itn,
        "chunk": chunk,
        "overlap": overlap,
    })
    return {"id": job.id}


@app.post("/api/jobs/{job_id}/retry")
def retry_job(job_id: str):
    """重试一个已结束的任务：复用已下载/已上传的媒体文件，跳过下载阶段。

    失败的任务（如显存不足、被取消）可以一键重来，不必重新下载视频。
    """
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "任务不存在")
    if job.status in ("queued", "running"):
        raise HTTPException(409, "任务进行中，请先取消")
    media = job.media_path
    options = dict(job.options)
    if media and Path(media).exists():
        # 已有本地媒体：直接复用，跳过下载阶段（重新转录不重新下载）
        new_job = _enqueue_job("local", media, options)
        return {"id": new_job.id, "reused_media": Path(media).name}
    if job.kind == "url":
        # 下载阶段就失败了：没有本地媒体，只能重新下载一次
        new_job = _enqueue_job("url", job.source, options)
        return {"id": new_job.id, "reused_media": None}
    raise HTTPException(400, "源媒体文件已不存在，无法直接重试（请重新提交链接或文件）")


@app.post("/api/history/retry")
def history_retry(name: str):
    """对历史记录重新转录：直接用 videos/ 里的源媒体文件，不重新下载。"""
    target = _resolve_history_file(name)
    stem = target.name[:-len("_out.txt")]
    exts = (TV.VIDEO_EXTENSIONS | TV.AUDIO_EXTENSIONS) if TV else set()
    media = None
    # 正常情况 stem 就是媒体文件名；重名结果（如 “xxx (2)_out.txt”）则去掉序号再找
    stems = [stem]
    m = re.match(r"^(.*) \(\d+\)$", stem)
    if m:
        stems.append(m.group(1))
    for s in stems:
        for ext in exts:
            cand = DOWNLOAD_DIR / (s + ext)
            if cand.exists() and cand.stat().st_size > 0:
                media = cand
                break
        if media is not None:
            break
    if media is None:
        raise HTTPException(400, "找不到对应的源媒体文件，无法重新转录")
    new_job = _enqueue_job("local", str(media), dict(DEFAULT_OPTIONS))
    return {"id": new_job.id, "reused_media": media.name}


@app.get("/api/jobs")
def list_jobs():
    return [j.to_dict() for j in sorted(JOBS.values(), key=lambda j: j.created, reverse=True)]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "任务不存在")
    return job.to_dict()


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "任务不存在")
    if job.status in ("done", "error", "cancelled"):
        return {"ok": True, "status": job.status}
    job.stop_event.set()
    if job.status == "queued":
        job.status, job.message = "cancelled", "已取消"
        job.finished = time.time()
    else:
        job.message = "取消中…"
    return {"ok": True, "status": job.status}


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "任务不存在")
    if job.status in ("queued", "running"):
        raise HTTPException(409, "任务进行中，请先取消")
    JOBS.pop(job_id, None)
    return {"ok": True}


@app.get("/api/history")
def history(q: str = ""):
    """历史记录列表；带 q 时按文件名或正文内容过滤（服务端搜索）。"""
    q = q.strip().lower()
    items = []
    exts = (TV.VIDEO_EXTENSIONS | TV.AUDIO_EXTENSIONS) if TV else set()
    for f in sorted(DOWNLOAD_DIR.glob("*_out.txt"),
                    key=lambda p: p.stat().st_mtime, reverse=True):
        title = f.name[:-len("_out.txt")]
        if q:
            try:
                body = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                body = ""
            if q not in title.lower() and q not in body.lower():
                continue
        media_name, media_size = None, None
        for ext in exts:
            cand = DOWNLOAD_DIR / (title + ext)
            if cand.exists():
                media_name, media_size = cand.name, cand.stat().st_size
                break
        st = f.stat()
        items.append({
            "name": f.name,
            "title": title,
            "size": st.st_size,
            "mtime": st.st_mtime,
            "media": media_name,
            "media_size": media_size,
        })
    return items


@app.get("/api/history/content")
def history_content(name: str):
    return PlainTextResponse(_resolve_history_file(name).read_text(encoding="utf-8"))


@app.delete("/api/history")
def history_delete(name: str):
    target = _resolve_history_file(name)
    target.unlink()
    # 顺带尝试删除同名媒体文件（可选，不存在则忽略）
    stem = target.name[:-len("_out.txt")]
    exts = (TV.VIDEO_EXTENSIONS | TV.AUDIO_EXTENSIONS) if TV else set()
    for ext in exts:
        cand = DOWNLOAD_DIR / (stem + ext)
        if cand.exists():
            try:
                cand.unlink()
            except OSError as e:
                print(f"[cleanup] 历史文件删除失败（已忽略）: {e}")
            break
    return {"ok": True}


@app.post("/api/open-folder")
def open_folder():
    if not hasattr(os, "startfile"):
        raise HTTPException(400, "仅支持 Windows")
    os.startfile(str(DOWNLOAD_DIR))  # noqa: S606 - 本地工具的预期行为
    return {"ok": True}


@app.post("/api/quit")
def quit_server():
    def _die():
        time.sleep(0.4)
        os._exit(0)
    threading.Thread(target=_die, daemon=True).start()
    return {"ok": True}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main():
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(errors="replace")

    parser = argparse.ArgumentParser(description="steno WebUI — 视频/音频转文字网页界面")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8321)
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    parser.add_argument("--no-preload", action="store_true",
                        help="启动后不在后台预加载模型（默认后台预加载，让第一个任务更快）")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/"
    print(f"steno WebUI 启动中: {url}")
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    # 后台预加载模型：不阻塞界面启动（torch 导入 ~10s + 权重加载）
    if not args.no_preload:
        threading.Thread(target=_preload_model_async, daemon=True).start()
    # CUDA 探测同样放后台（需要 import torch）
    threading.Thread(target=_warm_cuda_async, daemon=True).start()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
