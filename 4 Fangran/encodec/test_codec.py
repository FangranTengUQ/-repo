import wave
import time
import os
import tempfile
import subprocess
import numpy as np
import math
import torch
import torchaudio

from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Optional
from encodec import EncodecModel
from encodec.utils import convert_audio

# 定义编解码器接口
class CodecInterface(ABC):
    """音频编解码器的抽象接口类，定义必须实现的核心方法"""
    
    @abstractmethod
    def frame_processing(self, audio_data: np.ndarray) -> List[np.ndarray]:
        pass
    
    @abstractmethod
    def encode(self, frame: np.ndarray) -> bytes:
        pass
    
    @abstractmethod
    def decode(self, encoded_data: bytes) -> np.ndarray:
        pass
    
    @abstractmethod
    def combine_frames(self, decoded_frames: List[np.ndarray]) -> np.ndarray:
        pass


# 基础编解码器实现
class BaseCodec(CodecInterface):
    """基础编解码器类，提供通用结构，请补充这部分"""
    
    def __init__(self, frame_size: int= 80, device='cpu'):   # 80ms each frame
        """
        初始化基础编解码器
        
        参数:
            frame_size: 每帧的采样点数
            overlap: 帧之间的重叠采样点数
        """
        self.frame_size = frame_size
        self.device = device;
        self.model = EncodecModel.encodec_model_24khz().to(device)
        self.model.set_target_bandwidth(1.5)
        print("帧大小={frame_size}")
        
    def frame_processing(self, audio_data: torch.Tensor) -> List[torch.Tensor]:
        """将音频数据分帧"""    
        frames = []
        hop_size = self.frame_size*24  # 帧移大小
        audio_data_len = audio_data.shape[2]
        # 计算总帧数
        total_frames = math.ceil(audio_data_len / hop_size)
        
        for i in range(total_frames):
            start_idx = i * hop_size
            end_idx = start_idx + self.frame_size*24
            
            # 如果最后一帧不够长，用零填充
            if end_idx > audio_data_len:                
                frame = torch.zeros(audio_data.shape[0],audio_data.shape[1],self.frame_size*24, dtype=torch.float32)
                actual_length = audio_data_len - start_idx
                frame[:,:,:actual_length] = audio_data[:,:,start_idx:start_idx + actual_length]
            else:
                frame = audio_data[:,:,start_idx:end_idx]
            
            frames.append(frame)
        
        # 如果一帧都没有，至少保证有一帧
        if not frames:
            frames = [audio_data.copy()]
            
        return frames

    def encode(self, frame: torch.Tensor) -> bytes:
        """编码实现"""       
        with torch.no_grad():
            # 编码过程 
            encoded_data = self.model.encode(frame)
        # tensor->numpy
        encoded_data = encoded_data[0][0].flatten().numpy()
        
        # 10bit -> 8bit
        bit_buffer = 0
        bits_in_buffer = 0
        result_bytes = []
        
        for num in encoded_data:
            bit_buffer = (bit_buffer << 10) | num
            bits_in_buffer += 10
            
            while bits_in_buffer >= 8:
                byte_val = (bit_buffer >> (bits_in_buffer - 8)) & 0xFF
                result_bytes.append(byte_val)
                bits_in_buffer -= 8
                bit_buffer &= (1 << bits_in_buffer) - 1       
        if bits_in_buffer > 0:
            # 左移填充0，然后提取最后一个字节
            byte_val = (bit_buffer << (8 - bits_in_buffer)) & 0xFF
            result_bytes.append(byte_val)
        return bytes(result_bytes)

    def decode(self, encoded_data_bits: bytes) -> np.ndarray:
        """解码实现"""
        
        # 8bit -> 10bit
        bit_buffer = 0
        bits_in_buffer = 0
        result_10bit = []
    
        for byte_val in encoded_data_bits:
            # 将8bit字节放入缓冲区
            bit_buffer = (bit_buffer << 8) | byte_val
            bits_in_buffer += 8
        
            # 当缓冲区有至少10bit时，提取10bit数字
            while bits_in_buffer >= 10: #and len(result_10bit) < original_length:
                # 提取最高10bit
                num_10bit = (bit_buffer >> (bits_in_buffer - 10)) & 0x3FF
                result_10bit.append(num_10bit)
            
                # 更新缓冲区
                bits_in_buffer -= 10
                bit_buffer &= (1 << bits_in_buffer) - 1  # 清除已提取的位    
    
        # numpy->tensor
        rsp_size = math.ceil(self.frame_size*24/320)
        result_10bit_tensor = torch.tensor(result_10bit).reshape(1, len(result_10bit)//rsp_size, rsp_size)
        result_10bit_tensor = [(result_10bit_tensor, None)]

        with torch.no_grad():
            # 解码过程 
            decoded_frames = self.model.decode(result_10bit_tensor)
        return decoded_frames
    def combine_frames(self, decoded_frames:torch.Tensor) -> torch.Tensor:
        """将解码后的帧组合成完整音频"""       
        decoded_wav = torch.cat(decoded_frames,dim=2)
        return decoded_wav


# 测试函数
def test_codec(codec: CodecInterface, input_audio_path: str, output_audio_path: str) -> dict:
    """
    测试编解码器的性能
    
    参数:
        codec: 要测试的编解码器实例
        input_audio_path: 输入音频文件路径
        output_audio_path: 输出解码音频文件路径
        
    返回:
        测试结果字典，包含编码时间、解码时间、码流等信息
    """
    # 1. 读入音频
    print(f"读取音频文件: {input_audio_path}")
    
    with wave.open(input_audio_path, 'rb') as wf:
        params = wf.getparams()
        n_channels, sampwidth, framerate, n_frames = params[:4]
        audio_data = wf.readframes(n_frames)
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
    
    wav0, orig_sr = torchaudio.load(input_audio_path,format="wav")
    wav = convert_audio(wav0, orig_sr, codec.model.sample_rate, codec.model.channels) #24kHz and 1 channel
    wav = wav.unsqueeze(0).to(codec.device)  # 转换为Tensor
    
    # 2. 分帧处理
    print("进行分帧处理...")
    frames = codec.frame_processing(wav)
    total_frames = len(frames)
    print(f"分帧完成，共 {total_frames} 帧")
    
    # 3. 编码和解码过程
    encoded_data_list: List[bytes] = []
    decoded_frames: torch.Tensor = []
    total_encode_time = 0.0
    total_decode_time = 0.0
    
    print("开始编码和解码过程...")
    for i, frame in enumerate(frames):
        # 编码
        t1 = time.perf_counter()
        encoded_data = codec.encode(frame)
        encode_time = (time.perf_counter() - t1) * 1000  # 转换为毫秒
        total_encode_time += encode_time
        encoded_data_list.append(encoded_data)
        
        # 解码
        t2 = time.perf_counter()
        decoded_frame = codec.decode(encoded_data)
        decode_time = (time.perf_counter() - t2) * 1000  # 转换为毫秒
        total_decode_time += decode_time
        decoded_frames.append(decoded_frame)
    
    # 4. 合帧处理
    print("进行合帧处理...")
    decoded_audio = codec.combine_frames(decoded_frames)
    decoded_audio = decoded_audio.squeeze(0).cpu()

    # 确保解码后音频长度、采样率、存储格式与原始一致
    decoded_audio_out = decoded_audio[:,:wav.shape[2]]
    torchaudio.save(output_audio_path, decoded_audio_out, codec.model.sample_rate)    
    #wavout = convert_audio(decoded_audio_out, codec.model.sample_rate, orig_sr, 1)
    #audio_int16 = (wavout * 32767).clamp(-32768, 32767).to(torch.int16)
    #torchaudio.save(output_audio_path,audio_int16, orig_sr, encoding="PCM_S",bits_per_sample=16)
    # 5. 计算码流 (kbps)
    total_bits = sum(len(data)*8 for data in encoded_data_list) 
    audio_duration = len(wav0[0])/orig_sr  # 音频时长(秒)
    bitrate = (total_bits / 1024) / audio_duration  # kbps
    
    # 6. 保存解码后的语音
    print(f"保存解码后的音频到: {output_audio_path}")
    """
    with wave.open(output_audio_path, 'wb') as wf:
        wf.setparams(params)
        decoded_audio_int16 = np.int16(decoded_audio)
        wf.writeframes(decoded_audio_int16.tobytes())
    """
    # 输出测试结果
    print("\n===== 测试结果 =====")
    print(f"总编码时间: {total_encode_time:.2f} ms")
    print(f"平均每帧编码时间: {total_encode_time / total_frames:.4f} ms")
    print(f"总解码时间: {total_decode_time:.2f} ms")
    print(f"平均每帧解码时间: {total_decode_time / total_frames:.4f} ms")
    print(f"码流: {bitrate:.2f} kbps")
    print(f"原始音频时长: {audio_duration:.2f} 秒")
    print(f"原始音频大小: {wav0.numel() * 2 / 1024:.2f} KB")  # 假设16位采样
    print(f"编码后总大小: {sum(len(data) for data in encoded_data_list) / 1024:.2f} KB")
    
    return {
        "encode_time_ms": total_encode_time,
        "avg_encode_time_per_frame_ms": total_encode_time / total_frames,
        "decode_time_ms": total_decode_time,
        "avg_decode_time_per_frame_ms": total_decode_time / total_frames,
        "bitrate_kbps": bitrate,
        "decoded_audio_path": output_audio_path,
        "original_duration_sec": audio_duration,
        "total_frames": total_frames
    }


# 示例使用
if __name__ == "__main__":
    # 创建基础编解码器实例（使用内置的简单编码解码）
    basic_codec = BaseCodec(80) #80 ms for default
    
    # 运行测试
    test_results = test_codec(
        codec=basic_codec,
        input_audio_path="input.wav",
        output_audio_path="decoded_output.wav"
    )
