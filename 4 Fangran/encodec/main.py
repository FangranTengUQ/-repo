# -*- coding: utf-8 -*-
"""
Spyder 编辑器

这是一个临时脚本文件。
"""
import os
import sys
import torch
import torchaudio

from encodec import EncodecModel
from encodec.utils import convert_audio

# ----------------------------------
# 1. 环境配置
# ----------------------------------
# 安装必要依赖 (需提前执行)
# pip install torch torchaudio git+https://github.com/facebookresearch/encodec

# ----------------------------------
# 2. 初始化模型
# ----------------------------------
def load_model(device='cuda' if torch.cuda.is_available() else 'cpu'):
    # 选择支持的带宽（6kbps, 12kbps, 24kbps）
    model = EncodecModel.encodec_model_24khz().to(device)
    
    # 设置目标带宽（1.5 = 6kbps, 3.0 = 12kbps, 6.0 = 24kbps）
    model.set_target_bandwidth(6.0)
    
    return model

# ----------------------------------
# 3. 编码解码主流程
# ----------------------------------
def encode_decode_audio(input_path, output_path, sr=24000):
    # 初始化硬件设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(device)
    
    try:
        # 加载音频并进行预处理
        wav, orig_sr = torchaudio.load(input_path,format="wav")
        wav = convert_audio(wav, orig_sr, model.sample_rate, model.channels)
        wav = wav.unsqueeze(0).to(device)  # 添加batch维度
        
        # ⚠️ 必须启用评估模式
        with torch.no_grad():
            # 编码过程 (生成离散编码)
            encoded_frames = model.encode(wav)
            
            # 解码过程 (重建音频)
            reconstructed = model.decode(encoded_frames)
        
        # 保存结果
        reconstructed = reconstructed.squeeze(0).cpu()
        torchaudio.save(output_path, reconstructed, model.sample_rate)
        
        print(f"成功处理: {input_path} -> {output_path}")
        
    except RuntimeError as e:
        print(f"错误: {str(e)}")
        print("可能原因：1. 内存不足 2. 音频太长 3. 不支持的格式")

# ----------------------------------
# 4. 执行示例
# ----------------------------------
if __name__ == "__main__":
    input_file = "1.wav"    # 输入音频路径
    output_file = "1output.wav"  # 输出路径
    if os.path.exists(input_file):
        print("文件存在")
    else:
        print("文件不存在，请检查路径")
    # 处理音频（建议时长不超过10秒）
    encode_decode_audio(input_file, output_file)
