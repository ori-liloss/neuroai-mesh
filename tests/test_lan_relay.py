"""platforms/lan_relay.py: only private LAN addresses, bytes passed through unchanged (streaming too)."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "platforms"))
import lan_relay  # noqa: E402

passed = 0


def ok(cond, label):
    global passed
    if not cond:
        raise AssertionError(label)
    passed += 1
    print("ok  ", label)


ok(lan_relay.private_lan("192.168.1.20") and lan_relay.private_lan("10.0.0.5") and lan_relay.private_lan("172.20.1.1"),
   "private LAN addresses are accepted")
ok(not lan_relay.private_lan("0.0.0.0") and not lan_relay.private_lan("8.8.8.8") and not lan_relay.private_lan("172.32.0.1"),
   "0.0.0.0, public and out-of-range addresses are refused")
ok(not lan_relay.private_lan("127.0.0.1") and not lan_relay.private_lan("not-an-ip"), "loopback and junk are not 'private LAN'")
ok(not lan_relay.peer_ok("127.0.0.1", False) and lan_relay.peer_ok("127.0.0.1", True), "loopback peers only when allowed (tests)")
ok(not lan_relay.peer_ok("203.0.113.9", True), "a public peer is always dropped")


async def scenario():
    # A fake model that streams three chunks with pauses, like SSE.
    async def model(reader, writer):
        await reader.read(1024)
        for chunk in (b"data: one\n\n", b"data: two\n\n", b"data: [DONE]\n\n"):
            writer.write(chunk)
            await writer.drain()
            await asyncio.sleep(0.05)
        writer.close()

    upstream = await asyncio.start_server(model, "127.0.0.1", 0)
    up_port = upstream.sockets[0].getsockname()[1]
    ready = asyncio.get_running_loop().create_future()
    relay = asyncio.create_task(lan_relay.serve("127.0.0.1", 0, f"127.0.0.1:{up_port}", allow_loopback=True, ready=ready))
    port = await ready
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(b"POST /v1/chat/completions HTTP/1.1\r\n\r\n")
    await writer.drain()
    first = await asyncio.wait_for(reader.read(1024), 2)
    rest = await asyncio.wait_for(reader.read(), 2)
    writer.close()

    # The same relay with loopback not allowed drops a loopback client at once.
    ready2 = asyncio.get_running_loop().create_future()
    strict = asyncio.create_task(lan_relay.serve("127.0.0.1", 0, f"127.0.0.1:{up_port}", allow_loopback=False, ready=ready2))
    port2 = await ready2
    r2, w2 = await asyncio.open_connection("127.0.0.1", port2)
    try:
        w2.write(b"GET / HTTP/1.1\r\n\r\n")
        await w2.drain()
        dropped = await asyncio.wait_for(r2.read(), 2)
    except ConnectionError:  # Windows reports the drop as a reset, POSIX as an empty read
        dropped = b""
    for task in (relay, strict):
        task.cancel()
    upstream.close()
    return first, rest, dropped


first, rest, dropped = asyncio.run(scenario())
ok(first == b"data: one\n\n", "the first streamed chunk arrives on its own, before the rest (no buffering)")
ok(first + rest == b"data: one\n\ndata: two\n\ndata: [DONE]\n\n", "every byte passes through unchanged")
ok(dropped == b"", "a peer outside the home LAN gets nothing and is disconnected")

print(f"\nall {passed} checks passed")
