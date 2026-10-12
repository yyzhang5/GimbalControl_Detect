# SCI串口发送程序
# YYZ 2026-4-1

import struct
import serial
from Initial_SCi import initialize_serial_port   # 512行处调用

def sci_write(serial_port, data1, data2, control, verbose=False):
    """
    通过串口发送数据包
    
    参数:
    serial_port: 已初始化的串口对象
    data1: 第一个数据值 (float32，目标像素 u 坐标)
    data2: 第二个数据值 (float32，目标像素 v 坐标)
    control: 控制字 (uint8)，1=检测到目标，0=未检测到目标
    verbose: True 时打印发送的字节（仅调试用；正常运行保持 False，避免打印拖慢帧率）
    
    数据包格式（共 12 字节，多字节字段为小端序）:
    AA 55 [control] [data1 4字节] [data2 4字节] 0D

    与 DSP 端 cpu_timer2_isr 接收逻辑的对应关系:
    [0]  0xAA                     包头
    [1]  0x55                     包头
    [2]  control      (uint8)     控制字    -> ParB_RS422[0]
    [3]  data1 字节0 (LSB)                 -> ParB_RS422[1]
    [4]  data1 字节1                       -> ParB_RS422[2]
    [5]  data1 字节2                       -> ParB_RS422[3]
    [6]  data1 字节3 (MSB)                 -> ParB_RS422[4]
    [7]  data2 字节0 (LSB)                 -> ParB_RS422[5]
    [8]  data2 字节1                       -> ParB_RS422[6]
    [9]  data2 字节2                       -> ParB_RS422[7]
    [10] data2 字节3 (MSB)                 -> ParB_RS422[8]
    [11] 0x0D                     包尾/校验 -> ParB_RS422[9]

    说明:
          1) DSP 端用 *(float*)&dataBytes 还原数据，故必须发送
             IEEE-754 单精度 float32 小端字节序。
          2) 帧长必须恒为 12 字节（DSP 端 SCIFFRX.RXFFIL = 12，
             UartB_Rx_isr 每次固定取 12 字节），不要增删字节。
          3) control 目前 DSP 端未使用（cpu_timer2_isr 中
             Data2_C2TC1_W[2] = 1 为写死值），此处仅为后续扩展保留。
    """
    
    # 将控制字转换为8位无符号整数
    state_index = int(control) & 0xFF
    
    # 将数据转换为 float32 字节表示(小端序)
    # '<f' 表示小端序 IEEE-754 单精度浮点，与 DSP 端 *(float*)&dataBytes 对应
    d1_bytes = struct.pack('<f', float(data1))
    d2_bytes = struct.pack('<f', float(data2))
    
    # 构建发送数据包（共 12 字节，与 DSP 端映射一一对应）
    send_data = bytearray([
        0xAA,           # 包头 [0]
        0x55,           # 包头 [1]
        state_index,    # 控制字 [2]
        d1_bytes[0],    # data1 LSB [3]
        d1_bytes[1],    # data1     [4]
        d1_bytes[2],    # data1     [5]
        d1_bytes[3],    # data1 MSB [6]
        d2_bytes[0],    # data2 LSB [7]
        d2_bytes[1],    # data2     [8]
        d2_bytes[2],    # data2     [9]
        d2_bytes[3],    # data2 MSB [10]
        0x0D            # 包尾/校验 [11]
    ])

    # 通过串口发送（必须一次性写满 12 字节：
    #  DSP 端 SCIFFRX.RXFFIL = 12，按“12 字节”触发一次接收中断）
    try:
        written = serial_port.write(send_data)
        if written != len(send_data):
            print(f"发送数据不完整：期望 {len(send_data)} 字节，实际 {written} 字节")
            return False
        if verbose:
            print(f"成功发送数据包: {[hex(b) for b in send_data]}")
        return True
    except Exception as e:
        print(f"发送数据失败: {e}")
        return False
        
# 使用示例
if __name__ == "__main__":
    # 示例用法（需要先初始化串口）
    serial_port = initialize_serial_port(port='COM6', baudrate=115200)
    if serial_port:
        # 控制字按 DSP 约定取 1/0，不再使用无意义的示例值
        sci_write(serial_port, 320.0, 240.0, 1, verbose=True)
        serial_port.close()
        print("串口连接已关闭")




