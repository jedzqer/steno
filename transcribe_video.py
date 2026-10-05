import sys
import os
import subprocess
import tempfile
import argparse
from pathlib import Path

sys.path.insert(0, r"C:\Users\jed\SenseVoice\SenseVoice-official")
import re
import torch
import soundfile as sf
from model import SenseVoiceSmall
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

    clean = re.sub(r'[🎼]', '', "".join(all_texts))
    return clean


MODEL_DIR = r"C:\Users\jed\.cache\modelscope\hub\models\iic\SenseVoiceSmall"
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v'}
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.aac', '.m4a', '.wma'}


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
    parser = argparse.ArgumentParser(description='视频/音频转文字工具')
    parser.add_argument('input', help='输入视频或音频文件路径')
    parser.add_argument('-o', '--output', help='输出文本文件路径（可选）')
    parser.add_argument('--device', default='auto', help='推理设备：auto / cuda:0 / cpu')
    parser.add_argument('--language', default='zh', help='语言：auto / zh / en / yue / ja / ko')
    parser.add_argument('--no-itn', action='store_true', help='不启用智能标点与文字正则化')
    parser.add_argument('--ban-emo-unk', action='store_true', help='过滤 emo_unk 标记')
    parser.add_argument('--chunk', type=int, default=120, help='分块时长（秒）')
    parser.add_argument('--overlap', type=int, default=1, help='分块重叠（秒）')
    
    args = parser.parse_args()
    
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
        if cleanup_temp and os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)


if __name__ == "__main__":
    main()