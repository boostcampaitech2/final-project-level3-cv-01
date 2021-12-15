import time
import socket
import threading

from typing import Tuple

class ServerThread(threading.Thread):
    BUFFER_SIZE = 32768
    DECODER = 'utf-8'

    def __init__(self,
                 server: socket.socket,
                 client: socket.socket,
                 addr : Tuple[str, int]
                 ) -> None:
        threading.Thread.__init__(self)
        self.server = server
        self.client = client
        self.addr = addr
        self.cnt = 0

    def run(self) -> None:

        data = self.client.recv(1024)

        while True:

            if b'image' in data:
                self.cnt += 1
                st = time.time()

                with open(f'test.jpg', 'wb') as file:

                    while data:
                        data = self.client.recv(ServerThread.BUFFER_SIZE)
                        if b'image' in data:
                            code = data.decode(ServerThread.DECODER).strip()
                            print(f"code : {code}")
                            break
                        elif b'close' in data:
                            print(f"code : close")
                            self.client.close()
                            return

                        file.write(data)
                    print(f"Received images : {self.cnt:3d}")
                    print(f"Time spent : {time.time() - st:5.3f}s")

            elif b'connect' in data:
                code = data.decode(ServerThread.DECODER).strip()
                print(f"code : {code}")
                data = self.client.recv(ServerThread.BUFFER_SIZE)

            elif not data:
                print("Client closed abnormaly.")
                return
            
            else:
                print("Wrong data received.")
                return
                
