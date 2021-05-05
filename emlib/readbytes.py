from __future__ import annotations
import os
from struct import unpack, pack


def _num2bytes(n, typecode='i'):
    return pack('='+typecode, n)


def _bytes2num(s, typecode):
    return unpack('='+typecode, s)[0]


def read_i(f):
    "read an int"
    return unpack('=i', f.read(4))[0]


def read_f(f):
    "read a float"
    return unpack('=f', f.read(4))[0]


def read_d(f):
    "read a double"
    return unpack('=d', f.read(8))[0]

def read_ix(f, n):
    "read a number of ints"
    return unpack('=%di'%n, f.read(n*4))


def read_dx(f, n):
    "read multiple doubles"
    return unpack('=%dd'%n, f.read(n*8))


def read_x(f, b, n):
    """read a number of floats/doubles
    b: number of bits (4, 8)
    """
    typecode = {4:'f', 8:'d'}[b]
    return unpack('=%d%s'%(n,typecode), f.read(n*b))

def Int4(s, i=0):
    return unpack('=i', s[i:i+4])[0], i+4


def Float8(s, i=0):
    return unpack('=d', s[i:i+8])[0], i+8


def Float(b, s, i, n = 1):
    newi = i+b*n
    if b == 4:
        return unpack('=%df'%n, s[i:newi])
    if b == 8:
        return unpack('=%dd'%n, s[i:newi])


def read_sdif_1TRC(filename):
    # skip the headers: load the first 300 hundred characters and 
    # find the first occurence of 1TRC
    f = open(filename, 'rb')
    s = ''
    while 1:
        s = f.read(300)
        firsti = s.find('1TRC')
        if firsti != -1: break
    f.seek(0,0)
    end = os.path.getsize(filename)
    # discard the header
    f.seek( firsti, 1 )
    frames = []
    while f.tell() < end:
        f.seek(8, 1)
        time = read_d(f)
        f.seek(12, 1)
        datatype, rowCount, columnCount = read_ix(f, 3)
        if rowCount:
            frame = [read_x(f, datatype, columnCount) for i in range(rowCount)]
        else:
            frame = []
        frames.append((time, frame))
    f.close()
    return frames

