# simple_serial_comm.py
import time
import serial  # pip install pyserial

class SimpleSerialMotor:
    """아두이노와 간단히 각도 두 개(Tilt, Pan)를 주고받는 최소 버전."""
    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 1.0):
        # 포트 열기
        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        # 아두이노 리셋 대기
        time.sleep(2)

    def send(self, tilt_deg: int, pan_deg: int):
        """각도 전송 (예: 'T:90,P:120\\n')"""
        packet = f"T:{int(tilt_deg)},P:{int(pan_deg)}\n"
        self.ser.write(packet.encode("utf-8"))
        return self.ser.readline().decode("utf-8", errors="ignore").strip()

    def ping(self):
        """연결 확인 (선택사항)"""
        try:
            self.ser.write(b"PING\n")
            line = self.ser.readline().decode("utf-8", errors="ignore").strip()
            return line
        except Exception:
            return False

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
