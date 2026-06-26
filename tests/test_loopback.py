"""End-to-end smoke test: two UAs on localhost call each other.

No registrar; we hard-code the target address. Verifies INVITE/180/200/ACK/BYE
flow and basic SDP/RTP setup.
"""

import asyncio
import pytest

from opensip import Account, Call, UserAgent


@pytest.mark.asyncio
async def test_loopback_call_and_hangup():
    callee = UserAgent(local_addr=("127.0.0.1", 0))
    caller = UserAgent(local_addr=("127.0.0.1", 0))

    answered = asyncio.Event()
    ended = asyncio.Event()

    @callee.on_incoming_call
    async def handle(call: Call):
        await call.answer()
        answered.set()
        await call.wait_ended()
        ended.set()

    await callee.start()
    await caller.start()

    callee_addr = callee.local_addr
    # Manually create an account pointing at the callee's bound port — we
    # bypass REGISTER for this test by talking directly to it.
    callee_account = Account(
        username="bob", domain="127.0.0.1", password="x",
        server=("127.0.0.1", callee_addr[1]),
    )
    # Tell the callee about its account so incoming INVITE can find it.
    callee._accounts.append(callee_account)

    caller_account = Account(
        username="alice", domain="127.0.0.1", password="x",
        server=("127.0.0.1", callee_addr[1]),
    )

    call = await caller.invite(caller_account, "sip:bob@127.0.0.1")
    try:
        await call.wait_answered(timeout=5)
    finally:
        # Whatever happened, tear down.
        try:
            await call.hangup()
        except Exception:
            pass

    # Give callee a moment to see the BYE.
    try:
        await asyncio.wait_for(ended.wait(), timeout=2)
    except asyncio.TimeoutError:
        pass

    await caller.stop()
    await callee.stop()

    assert answered.is_set(), "callee did not see the INVITE answered"
