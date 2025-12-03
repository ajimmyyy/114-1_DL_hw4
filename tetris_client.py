import socket
import struct
from io import BytesIO
from PIL import Image
import numpy as np

class TetrisClient:
    def __init__(self, host="127.0.0.1", port=10612):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))

    def send_cmd(self, cmd: str):
        cmd = cmd.strip() + "\n"
        self.sock.sendall(cmd.encode())

    def recv_exact(self, size):
        """Receive exact bytes"""
        buf = b""
        while len(buf) < size:
            chunk = self.sock.recv(size - len(buf))
            if not chunk:
                raise ConnectionError("Connection closed by server.")
            buf += chunk
        return buf

    def get_state(self):
        raw_over = self.recv_exact(1)
        is_over = bool(raw_over[0])

        raw_lines = self.recv_exact(4)
        removed_lines = struct.unpack(">I", raw_lines)[0]

        raw_size = self.recv_exact(4)
        png_size = struct.unpack(">I", raw_size)[0]

        if png_size == 0:
            return is_over, removed_lines, np.zeros((200, 100, 3), dtype=np.uint8)

        png_bytes = self.recv_exact(png_size)

        img = Image.open(BytesIO(png_bytes))
        img = np.array(img)

        return is_over, removed_lines, img

    def start(self):
        self.send_cmd("start")

    def move(self, x):
        self.send_cmd(f"move {x}")

    def rotate(self, cw, n=1):
        self.send_cmd(f"rotate {cw} {n}")

    def drop(self):
        self.send_cmd("drop")

    def close(self):
        self.sock.close()
