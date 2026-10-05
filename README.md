# steno · 视频/音频速记工具

基于 [SenseVoiceSmall](https://github.com/FunAudioLLM/SenseVoice) 模型的速记工具，支持视频提取音频并转录为文字。

## 环境要求

- Python 3.8+
- FFmpeg（需添加到系统 PATH）
- CUDA GPU（推荐 4GB+ 显存）

## 安装依赖

```bash
pip install torch soundfile
```

## 使用方法

```bash
# 基本用法
python transcribe_video.py <输入文件路径>

# 指定输出文件
python transcribe_video.py <输入文件路径> -o <输出文件路径>
```

## 支持的格式

**视频：** mp4, avi, mkv, mov, wmv, flv, webm, m4v

**音频：** wav, mp3, flac, ogg, aac, m4a, wma

## 使用示例

```bash
python transcribe_video.py "videos/meeting.mp4"
python transcribe_video.py "audio.wav" -o "result.txt"
```

## 输出

- 转录结果保存为 `<原文件名>_out.txt`
- 自动过滤模型输出的表情符号（🎼 等）

## 技术说明

- **分块策略：** 120 秒/块，1 秒重叠。SenseVoiceSmall 的 CTC decoder 在约 120 秒内表现最佳——更短会产生切块边界噪音（语句断裂、重复），更长则对齐质量下降
- **显存占用：** 模型 234M 参数（FP32 ~934MB），120s 分块推理约占用 1.5GB，4GB 显存可正常运行
