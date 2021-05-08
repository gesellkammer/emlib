"""
Utility functions related to network (find a port, create a udp server, etc)
"""
from __future__ import annotations
import socket
from typing import Optional as Opt


def findport() -> int:
    """
    Find a free port (for UDP communication)

    Returns:
        the port number

    Raises XXX if no ports available
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    except Exception as e:
        s.close()
        raise e


def udpsocket() -> socket.socket:
    """
    Creates a UDP socket

    Returns:
        a socket

    Example::

        # send some data
        sock = udpsocket()
        sock.sendto(b"mymessage", ('192.168.1.3', 8888))
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return sock


def udpserver(port: int, addr='127.0.0.1') -> socket.socket:
    """
    Create a udp server

    Args:
        port: the port to listen to
        addr: the server address

    Example::

        # receive some data
        sock = udpserver(8888)
        while True:
            # bufsize = 1024
            data, addr = sock.recvfrom(1024)

    https://wiki.python.org/moin/UdpCommunication
    """
    sock = udpsocket()
    sock.bind((addr, port))
    return sock 
