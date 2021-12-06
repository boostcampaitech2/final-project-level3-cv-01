import socket

from server_thread import ServerThread
from typing import Tuple


class Server:
    def __init__(self,
                 address: Tuple[str, int]=('172.17.0.3', 6006)
                 ) -> None:
        self.host, self.port = address
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    def start(self) -> None:
        self.socket.bind((self.host, self.port))
        self.socket.listen(1)
        print("Server opened.")
        print("Waiting for connection.")

        while True:
            client_sock, addr = self.socket.accept()
            print(f"Connected by : {addr}")
            st = ServerThread(self.socket, client_sock, addr)
            st.start()
            
            return


def main() -> None:
    server = Server(('172.17.0.3', 6006))
    server.start()

if __name__ == "__main__":
    main()
