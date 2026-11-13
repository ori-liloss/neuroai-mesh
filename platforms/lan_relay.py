"""Make this computer's local model reachable to Neuro Router on the home network, without moving it.

llama-server stays on 127.0.0.1:8080, where the worker and its checks expect it. This relay listens on
the computer's PRIVATE LAN address only (10/8, 172.16/12, 192.168/16; never 0.0.0.0, never a public
address), on the same port, and passes each connection byte for byte to 127.0.0.1:8080. Streaming
answers pass through as they come. A peer that is not on the home LAN is dropped before any byte is
read. Standard library only.

    python3 platforms/lan_relay.py --listen 192.168.1.20 --port 8080 --target 127.0.0.1:8080
"""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import sys

PRIVATE = [ipaddress.ip_network(n) for n in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")]
LOOPBACK = ipaddress.ip_network("127.0.0.0/8")


def private_lan(host: str) -> bool:
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.version == 4 and any(ip in net for net in PRIVATE)


def peer_ok(host: str, allow_loopback: bool) -> bool:
    if private_lan(host):
        return True
    try:
        return allow_loopback and ipaddress.ip_address(host) in LOOPBACK
    except ValueError:
        return False


async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except (ConnectionError, OSError):
        pass
    finally:
        try:
            writer.close()
        except (ConnectionError, OSError):
            pass


def handler(target_host: str, target_port: int, allow_loopback: bool):
    async def handle(client_r: asyncio.StreamReader, client_w: asyncio.StreamWriter) -> None:
        peer = (client_w.get_extra_info("peername") or ("", 0))[0]
        if not peer_ok(str(peer), allow_loopback):
            client_w.close()
            return
        try:
            up_r, up_w = await asyncio.wait_for(asyncio.open_connection(target_host, target_port), 5)
        except (OSError, asyncio.TimeoutError):
            client_w.close()
            return
        await asyncio.gather(pipe(client_r, up_w), pipe(up_r, client_w))

    return handle


async def serve(listen: str, port: int, target: str, allow_loopback: bool = False, ready=None) -> None:
    target_host, _, target_port = target.rpartition(":")
    server = await asyncio.start_server(handler(target_host, int(target_port), allow_loopback), listen, port)
    if ready is not None:
        ready.set_result(server.sockets[0].getsockname()[1])
    async with server:
        await server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--listen", required=True, help="this computer's private LAN address")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--target", default="127.0.0.1:8080")
    args = parser.parse_args()
    if not private_lan(args.listen):
        sys.exit(f"{args.listen} is not a private LAN address (10/8, 172.16/12, 192.168/16); not listening.")
    host = args.target.rpartition(":")[0]
    if host not in {"127.0.0.1", "localhost"}:
        sys.exit("--target must be this computer's own model (127.0.0.1:PORT).")
    print(f"Model relay: http://{args.listen}:{args.port} -> http://{args.target} (home LAN only)", flush=True)
    try:
        asyncio.run(serve(args.listen, args.port, args.target))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
