#!/usr/bin/env python3
"""One-request latency: plain HuggingFace vs tiny-engine (same model, same prompt).

Measures TTFT, TPOT, e2e, tok/s for a single greedy generation.

    python scripts/compare_hf_vs_tiny_one_request.py
    python scripts/compare_hf_vs_tiny_one_request.py --max-tokens 64 --prompt "Explain KV cache in one sentence."

Notes:
  - One short warmup run per backend (excluded from metrics) so CUDA init isn't the story.
  - HF uses transformers generate + TextIteratorStreamer for TTFT.
  - tiny-engine runs twice: Stage-2 (no KV) and Stage-4/5-ish (paged + continuous).
"""

from __future__ import annotations

import argparse
import gc
import time
from dataclasses import dataclass
from threading import Thread

import torch


@dataclass
class Metrics:
    name: str
    prompt_tokens: int
    output_tokens: int
    ttft_ms: float
    tpot_ms: float | None
    e2e_ms: float
    tok_s: float
    text: str


def release():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def run_hf(model_id: str, prompt: str, max_tokens: int, warmup: bool) -> Metrics | None:
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda")
    model.eval()

    messages = [{"role": "user", "content": prompt}]
    encoded = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    input_ids = encoded["input_ids"].to(model.device)
    prompt_tokens = int(input_ids.shape[-1])
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    def once(measure: bool) -> Metrics | None:
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            do_sample=False,  # greedy
            streamer=streamer,
            pad_token_id=tok.eos_token_id,
        )
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask
        # For fair length compare with tiny ignore_eos: force exact max_tokens on measured run
        if measure:
            gen_kwargs["eos_token_id"] = None
            gen_kwargs["min_new_tokens"] = max_tokens
            gen_kwargs["max_new_tokens"] = max_tokens
        else:
            gen_kwargs["eos_token_id"] = tok.eos_token_id

        t0 = time.perf_counter()
        thread = Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()
        ttft = None
        chunks: list[str] = []
        for piece in streamer:
            if ttft is None:
                ttft = time.perf_counter() - t0
            chunks.append(piece)
        thread.join()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        e2e = time.perf_counter() - t0
        text = "".join(chunks)
        # Count new tokens via re-encode of generated text (approx) — better: use output length from generate
        # Streamer path: re-tokenize full text underestimates; use max_tokens when forced.
        out_tokens = max_tokens if measure else max(1, len(tok.encode(text, add_special_tokens=False)))
        if not measure:
            return None
        tpot = (e2e - ttft) / (out_tokens - 1) if out_tokens > 1 and ttft is not None else None
        return Metrics(
            name="HuggingFace (transformers generate + streamer)",
            prompt_tokens=prompt_tokens,
            output_tokens=out_tokens,
            ttft_ms=(ttft or 0) * 1000,
            tpot_ms=(tpot * 1000) if tpot is not None else None,
            e2e_ms=e2e * 1000,
            tok_s=out_tokens / e2e,
            text=text.strip().replace("\n", " ")[:120],
        )

    if warmup:
        _ = once(measure=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    m = once(measure=True)
    del model, tok
    release()
    return m


def run_tiny(model_id: str, prompt: str, max_tokens: int, features: list[str] | None, label: str, warmup: bool) -> Metrics:
    from tiny_engine import LLMEngine
    from tiny_engine.config import EngineConfig
    from tiny_engine.features import resolve_features

    values: dict = {"model": model_id, "device": "cuda", "dtype": "bfloat16", "record_steps": True}
    if features:
        _, settings = resolve_features(features)
        values.update(settings)
    config = EngineConfig(**values)
    config.validate()
    engine = LLMEngine(config)

    messages = [{"role": "user", "content": prompt}]
    ids = engine.encode_chat(messages)
    params = engine.sampling_params(max_tokens=max_tokens, temperature=0.0, ignore_eos=True)

    def once(measure: bool) -> Metrics | None:
        engine.step_log.clear()
        t0 = time.perf_counter()
        engine.add_request(list(ids), params)
        ttft = None
        final = None
        text_parts: list[str] = []
        while engine.has_unfinished_requests():
            for out in engine.step():
                if ttft is None:
                    ttft = time.perf_counter() - t0
                if out.delta_text:
                    text_parts.append(out.delta_text)
                if out.finished:
                    final = out
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        e2e = time.perf_counter() - t0
        if not measure:
            return None
        n = final.num_output_tokens
        tpot = (e2e - ttft) / (n - 1) if n > 1 and ttft is not None else None
        return Metrics(
            name=label,
            prompt_tokens=len(ids),
            output_tokens=n,
            ttft_ms=(ttft or 0) * 1000,
            tpot_ms=(tpot * 1000) if tpot is not None else None,
            e2e_ms=e2e * 1000,
            tok_s=n / e2e,
            text="".join(text_parts).strip().replace("\n", " ")[:120],
        )

    if warmup:
        # short warmup
        wparams = engine.sampling_params(max_tokens=8, temperature=0.0, ignore_eos=True)
        engine.add_request(list(ids)[: min(32, len(ids))], wparams)
        while engine.has_unfinished_requests():
            engine.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        engine.step_log.clear()

    m = once(measure=True)
    del engine
    release()
    assert m is not None
    return m


def print_row(m: Metrics):
    tpot = f"{m.tpot_ms:.2f}" if m.tpot_ms is not None else "-"
    print(
        f"{m.name:<48} | in={m.prompt_tokens:<4} out={m.output_tokens:<4} | "
        f"TTFT {m.ttft_ms:7.1f} ms | TPOT {tpot:>7} ms | "
        f"e2e {m.e2e_ms:7.1f} ms | {m.tok_s:5.1f} tok/s"
    )
    print(f"    text: {m.text!r}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--prompt", default="Explain what a KV cache is in one short sentence.")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--no-warmup", action="store_true")
    args = p.parse_args()
    warmup = not args.no_warmup

    print(f"model={args.model}")
    print(f"prompt={args.prompt!r}")
    print(f"max_tokens={args.max_tokens}  greedy  warmup={warmup}")
    print()

    rows: list[Metrics] = []

    print(">>> HuggingFace …")
    hf = run_hf(args.model, args.prompt, args.max_tokens, warmup=warmup)
    if hf:
        rows.append(hf)
        print_row(hf)
    print()

    print(">>> tiny-engine Stage 2 (no KV cache) …")
    t0 = run_tiny(args.model, args.prompt, args.max_tokens, features=None,
                  label="tiny-engine Stage 2 (kv=none)", warmup=warmup)
    rows.append(t0)
    print_row(t0)
    print()

    print(">>> tiny-engine paged + continuous …")
    t1 = run_tiny(args.model, args.prompt, args.max_tokens, features=["paged", "batching"],
                  label="tiny-engine (paged + continuous)", warmup=warmup)
    rows.append(t1)
    print_row(t1)
    print()

    print("=" * 100)
    print(f"{'backend':<48} | {'TTFT ms':>10} | {'TPOT ms':>10} | {'e2e ms':>10} | {'tok/s':>8}")
    print("-" * 100)
    for m in rows:
        tpot = f"{m.tpot_ms:.2f}" if m.tpot_ms is not None else "-"
        print(f"{m.name:<48} | {m.ttft_ms:10.1f} | {tpot:>10} | {m.e2e_ms:10.1f} | {m.tok_s:8.1f}")

    if len(rows) >= 2:
        base = rows[0]
        print()
        print("vs HuggingFace:")
        for m in rows[1:]:
            def ratio(a, b):
                return f"{a / b:.2f}×" if b else "-"
            print(
                f"  {m.name}: TTFT {ratio(m.ttft_ms, base.ttft_ms)} of HF, "
                f"TPOT {ratio(m.tpot_ms or 0, base.tpot_ms or 0)} of HF, "
                f"tok/s {ratio(m.tok_s, base.tok_s)} of HF"
            )


if __name__ == "__main__":
    main()
