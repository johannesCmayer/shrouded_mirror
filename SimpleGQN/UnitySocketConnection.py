import socket as soc
import time

class Sender:
    def __init__(self):
        try:
            self.socket = soc.socket(soc.AF_INET, soc.SOCK_STREAM)
            self.socket.setsockopt(soc.SOL_SOCKET, soc.SO_REUSEADDR, 1)
            self.socket.connect(('127.0.0.1', 5005))
            print('connected')
        except:
            self.socket.close()
            print('raise')
            raise

    def send(self, message):
        try:
            self.socket.send(message)
        except:
            self.socket.close()
            print('raise')
            raise

if __name__ == '__main__':
    sender = Sender()
    payload = bytes(''.join((('T',) * 128*128) + ('<END>',)), 'utf-8')
    while True:
        sender.send(payload)
    sender.socket.close()