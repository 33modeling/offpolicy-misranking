"""Length-framed messages and file descriptors over an inherited local socketpair."""
import array
import json
import os
import socket
import struct

LIMIT=4*1024*1024


def send(channel,value,fds=()):
    body=json.dumps(value,allow_nan=False).encode()
    if len(body)>LIMIT:raise ValueError('runtime IPC message too large')
    header=struct.pack('!Q',len(body))
    rights=[(socket.SOL_SOCKET,socket.SCM_RIGHTS,array.array('i',fds))] if fds else []
    sent=channel.sendmsg([header],rights)
    if sent<len(header):channel.sendall(header[sent:])
    channel.sendall(body)


def receive(channel):
    header,ancillary,flags,_=channel.recvmsg(8,socket.CMSG_SPACE(64*array.array('i').itemsize))
    fds=[]
    for level,kind,value in ancillary:
        if level==socket.SOL_SOCKET and kind==socket.SCM_RIGHTS:
            found=array.array('i');found.frombytes(value[:len(value)//found.itemsize*found.itemsize]);fds.extend(found)
    try:
        if flags&socket.MSG_CTRUNC:raise ValueError('truncated runtime IPC descriptors')
        if not header:return None,fds
        while len(header)<8:
            chunk=channel.recv(8-len(header))
            if not chunk:raise EOFError('truncated runtime IPC header')
            header+=chunk
        length=struct.unpack('!Q',header)[0]
        if length>LIMIT:raise ValueError('runtime IPC message too large')
        chunks=[];remaining=length
        while remaining:
            chunk=channel.recv(min(65536,remaining))
            if not chunk:raise EOFError('truncated runtime IPC body')
            chunks.append(chunk);remaining-=len(chunk)
        return json.loads(b''.join(chunks)),fds
    except BaseException:
        for fd in fds:os.close(fd)
        raise
