# SCI串口初始化程序
# YYZ 2025-12-10


import serial
import serial.tools.list_ports

# 校验方式名称（仅用于打印提示）
_PARITY_NAME = {
    serial.PARITY_NONE:  '无校验',
    serial.PARITY_EVEN:  '偶校验',
    serial.PARITY_ODD:   '奇校验',
    serial.PARITY_MARK:  'Mark校验',
    serial.PARITY_SPACE: 'Space校验',
}

def initialize_serial_port(port='COM6', baudrate=115200, timeout=1,
                           bytesize=serial.EIGHTBITS,
                           parity=serial.PARITY_EVEN,
                           stopbits=serial.STOPBITS_ONE):
    """
    初始化串口连接
    
    参数:
    port: 串口号 (如 'COM1', '/dev/ttyUSB0')
    baudrate: 波特率
    timeout: 超时时间(秒)
    bytesize: 数据位
    parity: 校验位（必须与 DSP 端 SCI_setConfig 的校验设置保持一致）
    stopbits: 停止位
    
    返回:
    serial.Serial对象
    """
    
    # 如果未指定端口，自动查找可用端口
    if port is None:
        ports = list(serial.tools.list_ports.comports())
        if len(ports) > 0:
            port = ports[0].device
            print(f"自动选择端口: {port}")
        else:
            raise Exception("未找到可用串口")
    
    try:
        # 创建串口对象
        ser = serial.Serial(
            port=port,                    # 修改：使用传入的port参数而非硬编码'COM6'
            baudrate=baudrate,            # 修改：使用传入的baudrate参数而非硬编码115200
            bytesize=bytesize,            # 数据位
            parity=parity,                # 校验位（须与 DSP 端一致）
            stopbits=stopbits,            # 停止位
            timeout=timeout               # 读取超时
        )
        
        print(f"串口 {port} 初始化成功")
        print(f"参数: 波特率={baudrate}, 数据位={bytesize}, "
              f"校验位={_PARITY_NAME.get(parity, parity)}, 停止位={stopbits}")
        
        return ser
        
    except serial.SerialException as e:
        print(f"串口初始化失败: {e}")
        return None

def close_serial_port(ser):
    """
    关闭串口连接
    """
    if ser and ser.is_open:
        ser.close()
        print("串口已关闭")

# 使用示例
if __name__ == "__main__":
    # 初始化串口
    serial_port = initialize_serial_port(port='COM6', baudrate=115200)  # 修改：显式传递参数
    
    if serial_port:
        # 检查串口状态
        print(f"串口是否打开: {serial_port.is_open}")
        
        # 可以在这里添加数据收发操作
        # 例如发送数据: serial_port.write(b'Hello World')
        # 例如接收数据: data = serial_port.readline()
        
        # 关闭串口
        # close_serial_port(serial_port)