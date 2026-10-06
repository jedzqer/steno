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
import shutil
import sys
import tempfile
import threading
import time
import traceback
import uuid
import webbrowser
from pathlib import Path

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
    created: float
    started: float | None
    finished: float | None

    def __init__(self, kind: str, source: str, options: dict):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind                  # 'upload' | 'url'
        self.source = source              # 上传文件的绝对路径 / 视频 URL
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


DOWNLOAD_ATTEMPTS = 3        # 下载总尝试次数（首次 + 2 次整体重试）
DOWNLOAD_RETRY_DELAY = 4     # 重试前的等待秒数


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

    try:
        for attempt in range(DOWNLOAD_ATTEMPTS):
            try:
                return TV.download_video(
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
                    raise
                job.message = (f"下载出错，{DOWNLOAD_RETRY_DELAY} 秒后重试"
                               f"（{attempt + 1}/{DOWNLOAD_ATTEMPTS - 1}）…")
                print(f"[download] 第 {attempt + 1}/{DOWNLOAD_ATTEMPTS} 次下载失败，"
                      f"{DOWNLOAD_RETRY_DELAY} 秒后重试: {e}")
                _sleep_cancellable(DOWNLOAD_RETRY_DELAY, job.stop_event)
                job.message = "下载中…"
        raise RuntimeError("下载失败：重试次数已用完")   # 防御：理论上循环内必 return 或 raise
    finally:
        if job.stop_event.is_set():
            raise _Cancelled()


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
        input_path = Path(job.source)
        if not input_path.exists():
            raise RuntimeError(f"文件不存在: {input_path}")
    _check_cancel(job)

    suffix = input_path.suffix.lower()
    if suffix not in TV.VIDEO_EXTENSIONS and suffix not in TV.AUDIO_EXTENSIONS:
        raise RuntimeError(f"不支持的文件格式 '{suffix or '(无后缀)'}'")

    _check_cancel(job)
    job.stage, job.message, job.progress = "load", "加载模型…", None
    model, kwargs = _ensure_model(job.options.get("device", "auto"))
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

        def on_progress(i, total, name):
            _check_cancel(job)
            job.progress = i / total if total else None
            job.message = f"转录中 · 第 {i}/{total} 段"

        try:
            text = TV.transcribe_audio(
                audio_path, model, kwargs,
                language=job.options.get("language", "zh"),
                use_itn=job.options.get("use_itn", True),
                ban_emo_unk=False,
                chunk_sec=int(job.options.get("chunk", 120)),
                overlap_sec=int(job.options.get("overlap", 1)),
                progress_cb=on_progress,
                stop_event=job.stop_event,
            )
        except TV.TranscriptionCancelled:
            raise _Cancelled() from None

        if not text.strip():
            raise RuntimeError("转录结果为空（音频可能没有可识别的语音）")

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
    except OSError:
        pass


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
        "cuda": bool(TV and __import__("torch").cuda.is_available()),
        "model_loaded": bool(_MODELS),
        "model_devices": sorted(_MODELS),
        "model_dir_exists": TV is not None and Path(TV.MODEL_DIR).exists(),
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "yt_dlp": yt_dlp_ok,
        "download_dir": str(DOWNLOAD_DIR),
    }


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

    job = Job(mode, source, {
        "audio_only": audio_only,
        "cookies": cookies,
        "language": language,
        "device": device,
        "use_itn": use_itn,
        "chunk": chunk,
        "overlap": overlap,
    })
    JOBS[job.id] = job
    JOB_QUEUE.put(job.id)
    return {"id": job.id}


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
            except OSError:
                pass
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
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/"
    print(f"steno WebUI 启动中: {url}")
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
