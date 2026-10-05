"""Schedulers (Stage 5) with real KV managers and fake requests — no model."""

import torch

from tiny_engine.cache import PagedKVManager
from tiny_engine.request import Request
from tiny_engine.sampling import SamplingParams
from tiny_engine.scheduler import ContinuousScheduler, FIFOScheduler, StaticBatchScheduler


class CharDecoder:
    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(65 + t % 26) for t in ids)


def req(rid, prompt_len, max_tokens=8):
    return Request(rid, list(range(1, prompt_len + 1)), SamplingParams(max_tokens=max_tokens), CharDecoder(),
                   {0}, max_model_len=4096, device=torch.device("cpu"))


def run_step(sched):
    """Pretend the model ran: advance computed tokens, append a token where a request reached its end."""
    out = sched.schedule()
    for s in out.scheduled:
        r = s.request
        r.num_computed_tokens += s.num_new_tokens
        if r.num_computed_tokens == r.num_tokens:
            r.append_token(5)
            if r.status.finished:
                sched.finish(r)
    return out


def test_fifo_runs_one_request_at_a_time():
    s = FIFOScheduler(PagedKVManager(100, 4), 4096, 8192)
    a, b = req("a", 10, 2), req("b", 10, 2)
    s.add(a)
    s.add(b)
    out = run_step(s)
    assert [x.request.request_id for x in out.scheduled] == ["a"]
    assert out.scheduled[0].num_new_tokens == 10  # whole prompt
    out = run_step(s)
    assert [x.request.request_id for x in out.scheduled] == ["a"] and out.scheduled[0].num_new_tokens == 1
    out = run_step(s)  # a finished last step; b starts
    assert [x.request.request_id for x in out.scheduled] == ["b"]


def test_continuous_admits_new_requests_mid_flight():
    s = ContinuousScheduler(PagedKVManager(100, 4), 4096, 8, 8192)
    a = req("a", 10, 5)
    s.add(a)
    run_step(s)
    b = req("b", 6, 5)
    s.add(b)
    out = run_step(s)
    got = {x.request.request_id: x.num_new_tokens for x in out.scheduled}
    assert got == {"a": 1, "b": 6}  # a decodes while b prefills in the same step


def test_chunked_prefill_respects_token_budget():
    s = ContinuousScheduler(PagedKVManager(100, 4), 4096, 8, 16, chunked_prefill=True)
    a = req("a", 40)
    s.add(a)
    chunks = [run_step(s).scheduled[0].num_new_tokens for _ in range(3)]
    assert chunks == [16, 16, 8]
    assert a.num_output_tokens == 1  # sampled only after the last chunk


def test_chunked_prefill_shares_steps_with_decodes():
    s = ContinuousScheduler(PagedKVManager(100, 4), 4096, 8, 16, chunked_prefill=True)
    a = req("a", 4, 50)
    s.add(a)
    run_step(s)
    s.add(req("b", 40))
    out = run_step(s)
    got = {x.request.request_id: x.num_new_tokens for x in out.scheduled}
    assert got == {"a": 1, "b": 15}  # decode first, prompt chunk fills the rest of the budget


def test_static_batch_waits_for_batch_to_drain():
    s = StaticBatchScheduler(PagedKVManager(100, 4), 4096, 2, 8192)
    a, b, c = req("a", 4, 1), req("b", 4, 3), req("c", 4, 1)
    for r in (a, b, c):
        s.add(r)
    out = run_step(s)
    assert {x.request.request_id for x in out.scheduled} == {"a", "b"}
    out = run_step(s)  # a is done; c must wait for b even though a seat is free
    assert {x.request.request_id for x in out.scheduled} == {"b"}
    run_step(s)
    out = run_step(s)
    assert {x.request.request_id for x in out.scheduled} == {"c"}


def test_preemption_when_kv_is_full():
    kv = PagedKVManager(num_blocks=4, block_size=4)  # 16 slots
    s = ContinuousScheduler(kv, 4096, 8, 8192)
    a, b = req("a", 8, 20), req("b", 7, 20)
    s.add(a)
    s.add(b)
    run_step(s)  # a: 2 blocks for 8 tokens, b: 2 blocks for 7 tokens → pool full
    assert kv.num_free_blocks() == 0
    out = run_step(s)  # a needs a 3rd block for token 9 → b (last admitted) is preempted
    assert [r.request_id for r in out.preempted] == ["b"]
    assert b.num_computed_tokens == 0 and s.waiting[0] is b
    assert not kv.has("b")


def test_max_num_seqs():
    s = ContinuousScheduler(PagedKVManager(100, 4), 4096, 2, 8192)
    for i in range(4):
        s.add(req(f"r{i}", 4))
    out = run_step(s)
    assert len(out.scheduled) == 2 and s.num_waiting == 2
