import cv2
import socket
import numpy as np

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

class TetrisClient:
    def __init__(self, host="127.0.0.1", port=10612):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))

    def send_cmd(self, cmd: str):
        cmd = cmd.strip() + "\n"
        self.sock.sendall(cmd.encode())

    def recv_exact(self, size):
        buf = b""
        while len(buf) < size:
            chunk = self.sock.recv(size - len(buf))
            if not chunk:
                raise ConnectionError("Connection closed by server.")
            buf += chunk
        return buf

    def recv_png(self, img_size):
        if img_size < 8:
            return self._resync_png()

        first8 = self.recv_exact(8)
        if first8 != PNG_MAGIC:
            return self._resync_png(prefetch=first8)

        body = self.recv_exact(img_size - 8)
        return first8 + body

    def _resync_png(self, prefetch=b""):
        data = prefetch
        while PNG_MAGIC not in data:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("Lost connection while resync PNG.")
            data += chunk

            if len(data) > 200000:
                data = data[-100000:]

        idx = data.index(PNG_MAGIC)

        return data[idx:]

    def get_state(self):
        is_game_over = (self.recv_exact(1) == b'\x01')
        removed_lines = int.from_bytes(self.recv_exact(4), 'big')
        holes = int.from_bytes(self.recv_exact(4), 'big')
        hight = int.from_bytes(self.recv_exact(4), 'big')
        bumpiness = int.from_bytes(self.recv_exact(4), 'big')
        pillar = int.from_bytes(self.recv_exact(4), 'big')
        y_pos = int.from_bytes(self.recv_exact(4), 'big')
        contact = int.from_bytes(self.recv_exact(4), 'big')
        img_size = int.from_bytes(self.recv_exact(4), 'big')
        
        img_png = self.recv_png(img_size)

        if img_size == 0:
            return is_game_over, removed_lines, np.zeros((200, 100, 3), dtype=np.uint8)
        
        nparr = np.frombuffer(img_png, np.uint8)
        np_image = cv2.imdecode(nparr, -1)

        return is_game_over, removed_lines, holes, hight, bumpiness, pillar, y_pos, contact, np_image

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

if __name__ == "__main__":
    from PIL import Image
    client = TetrisClient()
    client.start()
    is_game_over, removed_lines, holes, hight, bumpiness, pillar, y_pos, contact, np_image = client.get_state()
    img = Image.fromarray(np_image, 'RGB')
    img.show()
    print(is_game_over, removed_lines, holes, hight, bumpiness, pillar, y_pos, contact)

    client.drop()
    is_game_over, removed_lines, holes, hight, bumpiness, pillar, y_pos, contact, np_image = client.get_state()
    img = Image.fromarray(np_image, 'RGB')
    img.show()
    print(is_game_over, removed_lines, holes, hight, bumpiness, pillar, y_pos, contact)

    client.drop()
    is_game_over, removed_lines, holes, hight, bumpiness, pillar, y_pos, contact, np_image = client.get_state()
    img = Image.fromarray(np_image, 'RGB')
    img.show()
    print(is_game_over, removed_lines, holes, hight, bumpiness, pillar, y_pos, contact)

    print("Done")
