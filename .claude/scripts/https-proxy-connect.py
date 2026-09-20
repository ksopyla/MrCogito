#!/usr/bin/env python3
"""SSH ProxyCommand: tunnel to <host> <port> through the cloud session's HTTPS proxy.

ssh(1) does not read HTTPS_PROXY, so in a Claude Code cloud session a direct TCP
connection to a GPU server is simply dropped. This speaks the CONNECT verb on ssh's
behalf and then shuttles bytes.

Usage in SSH_CONFIG:

    Host odra
      ProxyCommand /path/to/https-proxy-connect.py %h %p

This does NOT bypass the network policy: the proxy still decides, and answers 403
for any host not on the environment's allowlist. The target must be added under
Network access -> Custom at claude.ai/code, and must accept SSH on a port the proxy
is willing to tunnel.

With no HTTPS_PROXY set it connects directly, so one SSH_CONFIG works both in the
cloud and on the laptop.

NOTE: stdin/stdout are the SSH channel. Never print to stdout; diagnostics go to
stderr, which ssh surfaces with -v.
"""
import base64
import os
import select
import socket
import sys
from urllib.parse import urlparse

TIMEOUT = 30
BUF = 65536


def fail(msg):
    sys.stderr.write("https-proxy-connect: %s\n" % msg)
    sys.exit(1)


def connect_via_proxy(host, port, proxy):
    u = urlparse(proxy if "://" in proxy else "http://" + proxy)
    if not u.hostname:
        fail("could not parse HTTPS_PROXY=%r" % proxy)
    try:
        s = socket.create_connection((u.hostname, u.port or 8080), timeout=TIMEOUT)
    except OSError as e:
        fail("cannot reach proxy %s: %s" % (proxy, e))

    req = "CONNECT %s:%d HTTP/1.1\r\nHost: %s:%d\r\n" % (host, port, host, port)
    if u.username:
        cred = base64.b64encode(
            ("%s:%s" % (u.username, u.password or "")).encode()
        ).decode()
        req += "Proxy-Authorization: Basic %s\r\n" % cred
    s.sendall((req + "\r\n").encode())

    # Read exactly the CONNECT response header; bytes after it belong to the tunnel.
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = s.recv(1)
        if not chunk:
            fail("proxy closed the connection during CONNECT")
        buf += chunk

    status = buf.split(b"\r\n", 1)[0].decode(errors="replace")
    if " 200" not in status:
        if " 403" in status or " 407" in status:
            fail(
                "proxy denied %s:%d (%s).\n"
                "  That host:port is not permitted by this environment's network\n"
                "  policy. Add the hostname under Network access -> Custom at\n"
                "  claude.ai/code. If the hostname is already allowlisted, the port\n"
                "  is likely the problem: the proxy may only tunnel 443.\n"
                "  This is a policy decision, not a transient error -- do not retry."
                % (host, port, status)
            )
        fail("proxy refused CONNECT to %s:%d (%s)" % (host, port, status))
    return s


def shuttle(s):
    """Relay between ssh (stdin/stdout) and the socket until either side closes."""
    sock_fd, in_fd, out_fd = s.fileno(), sys.stdin.fileno(), sys.stdout.fileno()
    s.setblocking(False)
    os.set_blocking(in_fd, False)
    watch = [sock_fd, in_fd]
    try:
        while True:
            readable, _, errored = select.select(watch, [], watch)
            if errored:
                break
            if sock_fd in readable:
                try:
                    data = s.recv(BUF)
                except BlockingIOError:
                    pass  # spurious readiness; never fabricate bytes
                else:
                    if not data:
                        break  # peer closed
                    os.write(out_fd, data)
            if in_fd in readable:
                try:
                    data = os.read(in_fd, BUF)
                except BlockingIOError:
                    continue
                if not data:
                    # stdin EOF: half-close so the peer sees it, then drain the
                    # socket. Closing here would truncate data still in flight.
                    try:
                        s.shutdown(socket.SHUT_WR)
                    except OSError:
                        pass
                    watch = [sock_fd]
                else:
                    s.sendall(data)
    finally:
        s.close()


def main():
    if len(sys.argv) != 3:
        fail("usage: https-proxy-connect.py <host> <port>")
    host = sys.argv[1]
    try:
        port = int(sys.argv[2])
    except ValueError:
        fail("port must be an integer, got %r" % sys.argv[2])

    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if proxy:
        s = connect_via_proxy(host, port, proxy)
    else:
        try:
            s = socket.create_connection((host, port), timeout=TIMEOUT)
        except OSError as e:
            fail("cannot reach %s:%d: %s" % (host, port, e))
    shuttle(s)


if __name__ == "__main__":
    main()
