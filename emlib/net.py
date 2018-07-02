import socket


def udpsocket():
    """
    creates a UDP socket

    To send a message:

    sock = udpsocket()
    sock.sendto(b"mymessage", ('192.168.1.3', 8888))
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return sock


def udpserver(port, ip='127.0.0.1'):
    """
    To receive data:

    sock = udpserver(8888)
    while True:
        # bufsize = 1024
        data, addr = sock.recvfrom(1024)

    https://wiki.python.org/moin/UdpCommunication
    """
    sock = udpsocket()
    sock.bind((ip, port))
    return sock 
