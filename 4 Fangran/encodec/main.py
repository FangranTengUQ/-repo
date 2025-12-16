# -*- coding: utf-8 -*-
import os
import sys
import torch
import torchaudio

from encodec import EncodecModel
from encodec.utils import convert_audio

# ----------------------------------
# 1. 初始化模型
# ----------------------------------
def load_model(device='cuda' if torch.cuda.is_available() else 'cpu'):
    # 加载 24kHz 模型，支持的带宽: [1.5, 3.0, 6.0, 12.0, 24.0] kbps
    model = EncodecModel.encodec_model_24khz().to(device)
    return model

# ----------------------------------
# 2. 批量测试不同带宽
# ----------------------------------
def test_bandwidths(input_path, output_dir="results"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(device)

    # 支持的带宽列表 (kbps)
    # 越低压缩率越高，越高音质越好
    bandwidths = [1.5, 3.0, 6.0, 12.0, 24.0]

    try:
        # 加载并预处理音频
        wav, orig_sr = torchaudio.load(input_path)
        # 转换为单声道(1, T) 和 24kHz
        wav = convert_audio(wav, orig_sr, model.sample_rate, model.channels)
        wav = wav.unsqueeze(0).to(device)  # [Batch, Channels, Time]

        print(f"处理音频: {input_path}")
        print(f"输入时长: {wav.shape[-1] / model.sample_rate:.2f} 秒")
        print("-" * 50)
        print(f"{'带宽(kbps)':<15} | {'压缩比(约)':<15} | {'失真度(MSE)':<20}")
        print("-" * 50)

        for bw in bandwidths:
            model.set_target_bandwidth(bw)

            with torch.no_grad():
                # 编码 + 解码
                encoded_frames = model.encode(wav)
                reconstructed = model.decode(encoded_frames)

            # 计算简单的均方误差 (MSE) 作为失真参考
            # 注意：Encodec 是有损压缩，MSE 不一定完全代表听感，但可以作为参考
            reconstructed = reconstructed.squeeze(0) # [C, T]
            input_wav = wav.squeeze(0)               # [C, T]

            # 裁剪到相同长度（通常是对齐的，但以防万一）
            min_len = min(reconstructed.shape[-1], input_wav.shape[-1])
            mse = torch.mean((input_wav[:, :min_len] - reconstructed[:, :min_len])**2).item()

            # 保存结果
            output_filename = f"output_{bw}kbps.wav"
            output_path = os.path.join(output_dir, output_filename)
            torchaudio.save(output_path, reconstructed.cpu(), model.sample_rate)

            # 估算压缩比:
            # 原始 (24000 Hz * 16 bit) = 384 kbps (如果是16bit PCM)
            # 实际上输入是 float32，但音频通常按 16bit 计算原始大小
            compression_ratio = 384.0 / bw

            print(f"{bw:<15.1f} | {compression_ratio:<15.1f} | {mse:<20.6f}")

        print("-" * 50)
        print(f"所有结果已保存在 {output_dir}/ 文件夹下。")
        print("请把这些文件发给 Mentor 对比听感，选择最平衡的带宽。")

    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    input_file = "1.wav"
    if len(sys.argv) > 1:
        input_file = sys.argv[1]

    if os.path.exists(input_file):
        test_bandwidths(input_file)
    else:
        print(f"文件 {input_file} 不存在，请检查路径或生成测试文件。")
