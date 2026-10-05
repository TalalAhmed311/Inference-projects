"""Stop conditions and streaming text, using a fake one-character-per-token decoder (no model needed)."""

import torch

from tiny_engine.request import Request
from tiny_engine.sampling import SamplingParams

EOS = 0


class CharDecoder:
    """Token id N decodes to chr(N); id 0 is the EOS special token."""

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(t) for t in token_ids if not (skip_special_tokens and t == EOS))


def make(**params):
    return Request("r", [65, 66], SamplingParams(**params), CharDecoder(), {EOS}, max_model_len=64,
                   device=torch.device("cpu"))


def feed(req, text):
    return [req.append_token(ord(c)) for c in text]


def test_max_tokens_finishes_with_length():
    req = make(max_tokens=3)
    outs = feed(req, "abc")
    assert [o.finished for o in outs] == [False, False, True]
    assert outs[-1].finish_reason == "length"
    assert "".join(o.delta_text for o in outs) == "abc" == outs[-1].text


def test_eos_finishes_with_stop_and_is_not_in_text():
    req = make(max_tokens=10)
    feed(req, "hi")
    out = req.append_token(EOS)
    assert out.finished and out.finish_reason == "stop"
    assert out.text == "hi"


def test_ignore_eos_keeps_generating():
    req = make(max_tokens=4, ignore_eos=True)
    feed(req, "h")
    assert not req.append_token(EOS).finished
    outs = feed(req, "ij")
    assert outs[-1].finish_reason == "length" and outs[-1].num_output_tokens == 4


def test_stop_string_is_removed_and_never_streamed():
    req = make(max_tokens=20, stop=["END"])
    outs = feed(req, "helloEND")
    streamed = "".join(o.delta_text for o in outs)
    assert outs[-1].finish_reason == "stop"
    assert streamed == "hello" == outs[-1].text
    # Partial "E" / "EN" were held back, never sent before the stop was recognised.
    assert all("E" not in o.delta_text for o in outs)


def test_holdback_releases_text_that_turns_out_not_to_be_a_stop():
    req = make(max_tokens=20, stop=["END"])
    outs = feed(req, "ENx")
    assert "".join(o.delta_text for o in outs) == "E"  # the last len("END") - 1 = 2 chars are always held
    outs += feed(req, "yz")
    assert "".join(o.delta_text for o in outs) == "ENx"


def test_stop_token_ids():
    req = make(max_tokens=20, stop_token_ids=[ord("!")])
    outs = feed(req, "ok!")
    assert outs[-1].finish_reason == "stop"


def test_min_tokens_blocks_eos_until_reached():
    req = make(max_tokens=10, min_tokens=2)
    assert req.blocked_token_ids() == [EOS]
    feed(req, "ab")
    assert req.blocked_token_ids() == []


def test_max_model_len_finishes_with_length():
    req = Request("r", list(range(65, 65 + 62)), SamplingParams(max_tokens=100), CharDecoder(), {EOS},
                  max_model_len=64, device=torch.device("cpu"))
    outs = feed(req, "ab")
    assert outs[-1].finished and outs[-1].finish_reason == "length"


def test_abort():
    req = make(max_tokens=10)
    feed(req, "a")
    out = req.abort()
    assert out.finished and out.finish_reason == "abort"
