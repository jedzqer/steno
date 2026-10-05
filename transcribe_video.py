import sys
import os
import subprocess
import tempfile
import argparse
from pathlib import Path

# SenseVoice 官方仓库的本地 checkout（项目根目录下），提供 model.py
sys.path.insert(0, str(Path(__file__).resolve().parent / "SenseVoice-official"))
import re
import torch
import soundfile as sf
from model import SenseVoiceSmall  # type: ignore  # 运行时经上方 sys.path 注入解析（外部 SenseVoice 目录），静态分析无法解析
from funasr.utils.postprocess_utils import rich_transcription_postprocess


def extract_audio_from_video(video_path, output_audio_path):
    """使用ffmpeg从视频中提取音频"""
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
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg错误: {e.stderr}")
        return False
    except FileNotFoundError:
        print("错误: 未找到ffmpeg，请确保已安装ffmpeg并添加到PATH环境变量中")
        return False


class TranscriptionCancelled(Exception):
    """转录被用户取消时抛出。"""
    pass


def transcribe_audio(audio_path, model, kwargs, language="zh", use_itn=True,
                     ban_emo_unk=False, chunk_sec=120, overlap_sec=1,
                     progress_cb=None, stop_event=None):
    """转录音频文件为文本。

    progress_cb(index, total, name): 每完成一段回调一次。
    stop_event(threading.Event): 置位时取消转录。
    """
    data, sr = sf.read(audio_path, dtype='float32')
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    waveform = torch.from_numpy(data)

    chunk_len = chunk_sec * sr
    stride = chunk_len - overlap_sec * sr

    segments = []
    for start in range(0, waveform.shape[0], stride):
        end = min(start + chunk_len, waveform.shape[0])
        segments.append(waveform[start:end])
        if end == waveform.shape[0]:
            break

    all_texts = []
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
        all_texts.append(rich_transcription_postprocess(res[0][0]["text"]))

    # 过滤 SenseVoice 输出的情绪/事件 emoji（funasr 会把 <|ANGRY|>、<|BGM|> 等标记转成😡🎼👏等），
    # 按 Unicode 表情符号区段整体过滤，比逐个枚举更稳
    clean = re.sub(r'[\U0001F000-\U0001FFFF\uFE0F]', '', "".join(all_texts))
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