import os
import time
import socket
import threading

from shutil import copyfile
from typing import Tuple

class ServerThread(threading.Thread):
    # BUFFER_SIZE = 32768 # 32768
    BUFFER_SIZE = 1400
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

        flag = 0

        data = self.client.recv(ServerThread.BUFFER_SIZE)
        st = 0
        success = False

        while True:

            if b'connect' in data:
                code = data.decode(ServerThread.DECODER).strip()
                print(f"code : {code}")
                data = self.client.recv(ServerThread.BUFFER_SIZE)


            elif b'close' in data:
                print(f"code : close")
                self.client.close()
                return

            elif not data:
                print(data)
                print("Client closed.")
                return

            else:                
                self.cnt += 1
                save_name = f'saved.jpg'
                test_name = f'test.jpg'

                print(f"Frequency  : {time.time() - st:5.3f}s")
                st = time.time()

                with open(save_name, 'wb') as file:
                    has_next = False
                    success = False

                    if has_next:
                        file.write(data)

                    while data:

                        data = self.client.recv(ServerThread.BUFFER_SIZE)

                        if b'EOFimage' in data:
                            print('EOFimage')
                            packets = data.split(b'EOFimage')

                            file.write(packets[0])
                            data = packets[-1]
                            if not data:
                                has_next = False
                                data = b'image'
                            else:
                                has_next = True

                            success = True
                            
                            break

                        elif b'image' in data:
                            print('image')
                            packets = data.split(b'image')
                            
                            file.write(packets[-1])
                            data = packets[-1]
                            if not data:
                                data = b'image'

                        elif b'EOF' in data:
                            print('EOF')
                            packets = data.split(b'EOF')

                            file.write(packets[0])

                            has_next = False
                            success = True
                            data = b'image'

                            break

                        else:
                            file.write(data)

                if success:
                    copyfile(save_name, test_name)
                    # os.rename('test.jpg', 'saved.jpg')
                    # print(f"Received images : {self.cnt:3d}")
                    print(f"Time spent : {time.time() - st:5.3f}s")
                    print("="*30)
                    success = False