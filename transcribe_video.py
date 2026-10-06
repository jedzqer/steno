import sys
import os
import subprocess
import tempfile
import threading
import time
import argparse
from pathlib import Path

# SenseVoice 官方仓库的本地 checkout（项目根目录下），提供 model.py
sys.path.insert(0, str(Path(__file__).resolve().parent / "SenseVoice-official"))
import re
import numpy as np
import torch
import soundfile as sf
from model import SenseVoiceSmall  # type: ignore  # 运行时经上方 sys.path 注入解析（外部 SenseVoice 目录），静态分析无法解析
from funasr.utils.postprocess_utils import rich_transcription_postprocess


class TranscriptionCancelled(Exception):
    """转录被用户取消时抛出。"""
    pass


def extract_audio_from_video(video_path, output_audio_path, stop_event=None):
    """使用 ffmpeg 从视频中提取音频（16kHz 单声道 pcm_s16le）。

    stop_event: 可选 threading.Event。置位时立即 kill ffmpeg 并抛出
    TranscriptionCancelled，让 WebUI 的“提取音频”阶段可以即时取消。
    """
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vn',
        '-acodec', 'pcm_s16le',
        '-ar', '16000',
        '-ac', '1',
        '-y',
        output_audio_path
    ]

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE,
                                text=True, encoding='utf-8', errors='replace')
    except FileNotFoundError:
        print("错误: 未找到ffmpeg，请确保已安装ffmpeg并添加到PATH环境变量中")
        return False

    # 后台线程持续排空 stderr，避免管道缓冲区写满导致 ffmpeg 卡死
    stderr_chunks = []

    def _drain(stream):
        try:
            stderr_chunks.append(stream.read())
        except Exception:
            pass

    stderr_pipe = proc.stderr
    drain = None
    if stderr_pipe is not None:      # stderr=PIPE 启动，必定存在；仅为类型收窄
        drain = threading.Thread(target=_drain, args=(stderr_pipe,), daemon=True)
        drain.start()
    try:
        ret = None
        while True:
            ret = proc.poll()
            if ret is not None:
                break
            if stop_event is not None and stop_event.is_set():
                proc.kill()
                proc.wait()
                raise TranscriptionCancelled("音频提取已被用户取消")
            time.sleep(0.1)
        if drain is not None:
            drain.join(timeout=2)
        if ret != 0:
            print(f"ffmpeg错误: {''.join(stderr_chunks)[:2000]}")
            return False
        return True
    finally:
        if proc.poll() is None:      # 防御：任何异常路径都确保子进程被回收
            proc.kill()
        try:
            proc.wait(timeout=5)
        except Exception:
            pass


SENSEVOICE_SR = 16000   # SenseVoiceSmall 模型要求的音频采样率


def decode_audio_with_ffmpeg(audio_path, sr=SENSEVOICE_SR):
    """用 ffmpeg 将任意音频/视频解码为 sr 采样率、单声道的 float32 PCM 波形。

    libsndfile（soundfile 底层）不支持 AAC/M4A 等 yt-dlp 常见下载格式
    （sf.read 会报 Format not recognised），此类文件必须经 ffmpeg 解码；
    同时统一重采样到模型要求的 16kHz。返回 (numpy.float32 波形, 采样率)。
    """
    cmd = [
        'ffmpeg', '-nostdin', '-v', 'error',
        '-i', str(audio_path),
        '-vn',
        '-acodec', 'pcm_f32le',
        '-ac', '1',
        '-ar', str(sr),
        '-f', 'f32le', 'pipe:1',
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, check=True)
    except FileNotFoundError as e:
        raise RuntimeError("错误: 未找到ffmpeg，请确保已安装ffmpeg并添加到PATH环境变量中") from e
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or b'').decode(errors='replace').strip()
        raise RuntimeError(f"ffmpeg 解码音频失败: {stderr}") from e
    data = np.frombuffer(result.stdout, dtype=np.float32).copy()
    if data.size == 0:
        raise RuntimeError(f"ffmpeg 未能从 '{audio_path}' 解出任何音频数据")
    return data, sr


_PUNCT_RE = re.compile(r'[\s\W_]+', re.UNICODE)


def _significant_text(s):
    """去掉空白与标点后的有效字符序列，用于跨分块重叠区比对。"""
    return _PUNCT_RE.sub('', s)


def _merge_overlapped(prev_text, next_text, max_overlap_chars):
    """合并相邻分块的转录文本，剪掉后块头部与前块尾部重复的重叠区内容。

    分块按 stride 滑动时，相邻两块有 overlap_sec 秒的重叠音频，同一段话会被
    转录两次，直接拼接会在边界处出现重复字词。这里将两侧文本做“去空白/标点”
    归一化后，寻找最长的 尾部==头部 重复（至少 2 个有效字符，避免单字巧合），
    再映射回原文定位截断点；找不到匹配就原样拼接——宁可重复，也不误删。
    """
    if max_overlap_chars <= 0 or not prev_text or not next_text:
        return prev_text + next_text
    p, c = _significant_text(prev_text), _significant_text(next_text)
    limit = min(len(p), len(c), max_overlap_chars)
    matched = 0
    for k in range(limit, 1, -1):
        if p.endswith(c[:k]):
            matched = k
            break
    if not matched:
        return prev_text + next_text
    count = 0
    for idx, ch in enumerate(next_text):
        if not _PUNCT_RE.fullmatch(ch):
            count += 1
            if count == matched:
                rest = re.sub(r'^[\s\W_]+', '', next_text[idx + 1:])
                return prev_text + rest
    return prev_text + next_text


def transcribe_audio(audio_path, model, kwargs, language="zh", use_itn=True,
                     ban_emo_unk=False, chunk_sec=120, overlap_sec=1,
                     progress_cb=None, stop_event=None):
    """转录音频文件为文本。

    分块按 stride 滑动、相邻块带 overlap_sec 重叠：重叠区语音被转录两次，
    拼接时按“去标点/空白后的最长尾首重复”剪掉后块头部，避免边界处重复字词。

    progress_cb(index, total, name): 每完成一段回调一次。
    stop_event(threading.Event): 置位时取消转录。
    """
    # libsndfile 不支持 m4a/aac 等格式（报 Format not recognised），且模型要求 16kHz：
    # soundfile 读不出、或采样率非 16kHz 的音频统一回落到 ffmpeg 解码 + 重采样
    data = sr = None
    try:
        data, sr = sf.read(audio_path, dtype='float32')
        if len(data.shape) > 1:
            data = data.mean(axis=1)
    except RuntimeError:            # soundfile 的 LibsndfileError 等均继承 RuntimeError
        data = sr = None
    if data is None or sr != SENSEVOICE_SR:
        data, sr = decode_audio_with_ffmpeg(audio_path)
    waveform = torch.from_numpy(data)

    chunk_len = chunk_sec * sr
    stride = chunk_len - overlap_sec * sr

    segments = []
    for start in range(0, waveform.shape[0], stride):
        end = min(start + chunk_len, waveform.shape[0])
        segments.append(waveform[start:end])
        if end == waveform.shape[0]:
            break

    max_overlap_chars = 0 if overlap_sec <= 0 else min(400, overlap_sec * 20 + 10)
    merged = ''
    name = Path(audio_path).stem
    for i, seg in enumerate(segments):
        if stop_event is not None and stop_event.is_set():
            raise TranscriptionCancelled()
        print(f"[{name}] 处理第 {i+1}/{len(segments)} 段...")
        if progress_cb:
            progress_cb(i + 1, len(segments), name)
        with torch.no_grad():
            res = model.inference(
                data_in=seg.unsqueeze(0),
                data_lengths=torch.tensor([seg.shape[0]]),
                language=language,
                use_itn=use_itn,
                ban_emo_unk=ban_emo_unk,
                **kwargs,
            )
        merged = _merge_overlapped(merged, rich_transcription_postprocess(res[0][0]["text"]),
                                   max_overlap_chars)

    # 过滤 SenseVoice 输出的情绪/事件 emoji（funasr 会把 <|ANGRY|>、<|BGM|> 等标记转成😡🎼👏等），
    # 按 Unicode 表情符号区段整体过滤，比逐个枚举更稳
    clean = re.sub(r'[\U0001F000-\U0001FFFF\uFE0F]', '', merged)
    return clean


MODEL_DIR = r"C:\Users\jed\.cache\modelscope\hub\models\iic\SenseVoiceSmall"
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v'}
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.aac', '.m4a', '.wma'}
DOWNLOAD_DIR = Path(__file__).resolve().parent / 'videos'


def is_url(text):
    """判断输入是否为视频网站链接。"""
    return text.startswith(('http://', 'https://', 'www.'))


def download_video(url, output_dir=DOWNLOAD_DIR, audio_only=False,
                   cookies_from_browser=None, progress_cb=None):
    """用 yt-dlp 从视频网站（YouTube、B站等数千个站点）下载视频/音频。

    返回下载后的文件路径（Path）。文件保存到 output_dir（默认项目 videos/ 目录），
    文件名格式: 标题 [视频ID].扩展名。

    audio_only: 仅下载音轨，转录场景下载更快、占盘更小。
    cookies_from_browser: 浏览器名称（chrome/firefox/edge 等），
        用于需要登录的站点（如会员/ age-restricted 内容）。
    progress_cb: 可选回调 progress_cb(status, downloaded, total, speed)，
        status 为 'downloading'/'finished'；供 WebUI 展示进度。回调内抛出的
        异常会原样传播（WebUI 借此用 DownloadCancelled 中断下载）。
    """
    try:
        import yt_dlp
    except ImportError:
        print("错误: 未安装 yt-dlp，请先运行: pip install yt-dlp")
        sys.exit(1)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def progress_hook(d):
        if progress_cb is not None:   # WebUI 接管进度展示，控制台不再打印
            if d['status'] == 'downloading':
                progress_cb('downloading',
                            d.get('downloaded_bytes', 0),
                            d.get('total_bytes') or d.get('total_bytes_estimate'),
                            d.get('speed'))
            elif d['status'] == 'finished':
                progress_cb('finished', None, None, None)
            return
        if d['status'] == 'downloading':
            downloaded = d.get('downloaded_bytes', 0)
            total = d.get('total_bytes') or d.get('total_bytes_estimate')
            speed = d.get('speed')
            speed_str = f" {speed / 1048576:.1f} MB/s" if speed else ""
            if total:
                print(f"\r下载中: {downloaded / 1048576:.1f}/{total / 1048576:.1f} MB"
                      f"（{downloaded / total * 100:.1f}%）{speed_str}   ",
                      end='', flush=True)
            else:
                print(f"\r下载中: {downloaded / 1048576:.1f} MB{speed_str}   ",
                      end='', flush=True)
        elif d['status'] == 'finished':
            print("\n下载完成，正在合并/后处理...")

    ydl_opts = {
        'format': ('bestaudio/best' if audio_only else
                   'bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best'),
        'outtmpl': str(output_dir / '%(title).150s [%(id)s].%(ext)s'),
        'windowsfilenames': True,   # Windows 下使用更保守的文件名清洗规则
        'noplaylist': True,         # 传播放列表链接时只下载单个视频
        'quiet': True,
        'no_warnings': True,
        'noprogress': True,         # 关闭 yt-dlp 自带进度条，用自定义 progress_hook
        'progress_hooks': [progress_hook],
        # Python API 不会继承 yt-dlp CLI 的默认参数：不显式设置时下载重试次数为 0，
        # B 站 PCDN 节点（*.mcdn.bilivideo.cn）一次读超时就会让整个下载直接失败。
        'retries': 10,              # 网络错误重试次数；配合 continuedl（默认开启）断点续传
        'fragment_retries': 10,     # HLS/DASH 分片重试次数
        'file_access_retries': 5,   # 文件被占用/读写错误重试次数
        'extractor_retries': 3,     # 抽取器已知错误重试次数
        'socket_timeout': 20,       # 读超时秒数（显式声明，避免依赖 networking 层隐式默认值）
    }
    if not audio_only:
        ydl_opts['merge_output_format'] = 'mp4'
    if cookies_from_browser:
        ydl_opts['cookiesfrombrowser'] = (cookies_from_browser,)

    print("正在获取视频信息...")
    from yt_dlp.utils import DownloadError

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:  # type: ignore  # 官方 README 即以普通 dict 传参；_Params 为私有 TypedDict，无法静态构造
            info = ydl.extract_info(url, download=True)
            # requested_downloads[0]['filepath'] 是合并/后处理后的最终文件路径
            filepath = info.get('requested_downloads', [{}])[0].get('filepath')
            if not filepath:
                filepath = ydl.prepare_filename(info)
    except DownloadError as e:
        raise RuntimeError(f"yt-dlp 下载失败: {e}") from e

    print(f"下载完成: {filepath}")
    return Path(filepath)


def resolve_device(device="auto"):
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device


def load_model(model_dir=MODEL_DIR, device="auto"):
    device = resolve_device(device)
    m, kwargs = SenseVoiceSmall.from_pretrained(model=model_dir, device=device)
    m.eval()
    return m, kwargs


def main():
    # 重定向/管道输出时 Windows 默认 GBK 编不了 emoji，降级为替换而非崩溃
    # （reconfigure 只在 io.TextIOWrapper 上存在，用 getattr 兼容存根类型）
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, 'reconfigure', None)
        if callable(reconfigure):
            reconfigure(errors='replace')

    parser = argparse.ArgumentParser(description='视频/音频转文字工具，支持直接传入视频网站 URL 自动下载')
    parser.add_argument('input', help='输入视频/音频文件路径，或视频网站 URL（http(s):// 或 www. 开头）')
    parser.add_argument('--audio-only', action='store_true',
                        help='URL 下载时仅下载音轨（转录只需音频，下载更快）')
    parser.add_argument('--cookies-from-browser', default=None, metavar='BROWSER',
                        help='下载时从指定浏览器读取 cookies（chrome/firefox/edge 等，用于需登录的站点）')
    parser.add_argument('-o', '--output', help='输出文本文件路径（可选）')
    parser.add_argument('--device', default='auto', help='推理设备：auto / cuda:0 / cpu')
    parser.add_argument('--language', default='zh', help='语言：auto / zh / en / yue / ja / ko')
    parser.add_argument('--no-itn', action='store_true', help='不启用智能标点与文字正则化')
    parser.add_argument('--ban-emo-unk', action='store_true', help='过滤 emo_unk 标记')
    parser.add_argument('--chunk', type=int, default=120, help='分块时长（秒）')
    parser.add_argument('--overlap', type=int, default=1, help='分块重叠（秒）')
    
    args = parser.parse_args()

    if is_url(args.input):
        try:
            input_path = download_video(
                args.input,
                audio_only=args.audio_only,
                cookies_from_browser=args.cookies_from_browser,
            )
        except RuntimeError as e:
            print(f"\n错误: {e}")
            sys.exit(1)
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"错误: 文件 '{input_path}' 不存在")
            sys.exit(1)
    
    is_video = input_path.suffix.lower() in VIDEO_EXTENSIONS
    is_audio = input_path.suffix.lower() in AUDIO_EXTENSIONS
    
    if not is_video and not is_audio:
        print(f"错误: 不支持的文件格式 '{input_path.suffix}'")
        print(f"支持的视频格式: {', '.join(VIDEO_EXTENSIONS)}")
        print(f"支持的音频格式: {', '.join(AUDIO_EXTENSIONS)}")
        sys.exit(1)
    
    device = resolve_device(args.device)
    print(f"正在加载模型（{device}）...")
    m, kwargs = load_model(device=device)
    
    temp_audio_path = None
    cleanup_temp = False
    if is_video:
        print(f"正在从视频中提取音频: {input_path}")
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
        
        if not extract_audio_from_video(str(input_path), temp_audio_path):
            sys.exit(1)
        
        audio_path = temp_audio_path
        cleanup_temp = True
    else:
        audio_path = str(input_path)
        cleanup_temp = False
    
    try:
        print(f"正在转录音频: {audio_path}")
        text = transcribe_audio(
            audio_path, m, kwargs,
            language=args.language,
            use_itn=not args.no_itn,
            ban_emo_unk=args.ban_emo_unk,
            chunk_sec=args.chunk,
            overlap_sec=args.overlap,
        )
        
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = input_path.with_name(f"{input_path.stem}_out.txt")
        
        output_path.write_text(text, encoding="utf-8")
        print(f"\n转录完成！结果已保存到: {output_path}")
        print(f"\n转录内容:\n{text}")
        
    finally:
        if cleanup_temp and temp_audio_path and os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)


if __name__ == "__main__":
    main()