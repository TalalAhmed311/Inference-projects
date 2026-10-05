"""tiny-engine: the command-line front end.

    tiny-engine                              interactive menu: tick features, pick a model, choose what to run
    tiny-engine features                     list every feature and what it does
    tiny-engine config   [engine options]    show the resolved configuration (no model is loaded)
    tiny-engine chat     [engine options]    chat in the terminal, with per-reply metrics
    tiny-engine generate [engine options] "prompt" ["prompt" ...]
    tiny-engine serve    [engine options] [--port 8001]
    tiny-engine bench    STAGE [benchmark options]      STAGE = 2 | 3 | 4 | 5 | 6 | 7 | 8 | online

Engine options are the same everywhere; the main one is --features:

    tiny-engine chat --features paged,batching,prefix
    tiny-engine chat --features all --quantization none
    tiny-engine serve --features spec --num-speculative-tokens 6 --port 8001
    tiny-engine bench 5 --features prefix --rate 8

Also available as `python -m tiny_engine ...`.
"""

from __future__ import annotations

import argparse
import logging
import runpy
import shlex
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from tiny_engine.cli import add_engine_args, config_from_args, config_values, enabled_features
from tiny_engine.config import EngineConfig
from tiny_engine.features import ALIASES, DEFAULT_DRAFT_MODEL, FEATURES, features_table, parse_feature_list, resolve_features

STAGE_DIR = Path(__file__).resolve().parent.parent
BENCH_DIR = STAGE_DIR / "benchmarks"
BENCHMARKS = {
    "2": ("bench_offline.py", ["--kv-caches", "none"], "single request, no KV cache"),
    "3": ("bench_offline.py", ["--kv-caches", "none", "contiguous"], "no cache vs KV cache, single request"),
    "4": ("stage4_paged_capacity.py", [], "contiguous vs paged under the same KV memory"),
    "5": ("stage5_scheduling.py", [], "fifo vs static vs continuous vs chunked, Poisson arrivals"),
    "6": ("stage6_prefix_caching.py", [], "shared system prompt with and without prefix caching"),
    "7": ("stage7_speculative.py", [], "speculative decoding: k, batch size, temperature"),
    "8": ("stage8_quantization.py", [], "bf16 vs int8 vs int4 vs fp8: memory, speed, quality"),
    "online": ("run_online.sh", [], "Stage 1 bench.py against a running tiny-engine server"),
}


# ----------------------------------------------------------------------------- shared helpers


def add_sampling_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("sampling (defaults: the model's generation_config.json)")
    g.add_argument("--max-tokens", type=int, default=512)
    g.add_argument("--temperature", type=float, default=None)
    g.add_argument("--top-p", type=float, default=None)
    g.add_argument("--top-k", type=int, default=None)
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("--stop", action="append", default=None)
    g.add_argument("--ignore-eos", action="store_true")


def sampling_overrides(args) -> dict:
    return {"max_tokens": args.max_tokens, "temperature": args.temperature, "top_p": args.top_p,
            "top_k": args.top_k, "seed": args.seed, "stop": args.stop, "ignore_eos": args.ignore_eos or None}


def build_engine(args, **overrides):
    from tiny_engine import LLMEngine

    config = config_from_args(args, **overrides)
    print_features(config)
    print("loading model…", flush=True)
    return LLMEngine(config)


def print_features(config: EngineConfig) -> None:
    print(f"\n  model     {config.model}")
    for line in enabled_features(config):
        print(f"  ✓ {line}")
    print()


def stream_request(engine, prompt_ids: list[int], params) -> tuple[str, dict]:
    """Run one request, printing text as it streams. Ctrl-C stops it early."""
    before = (engine.stats.spec_draft_tokens, engine.stats.spec_accepted_tokens)
    t0 = time.perf_counter()
    rid = engine.add_request(prompt_ids, params)
    ttft, final, pieces = None, None, []
    try:
        while engine.has_unfinished_requests():
            for out in engine.step():
                if ttft is None:
                    ttft = time.perf_counter() - t0
                if out.delta_text:
                    sys.stdout.write(out.delta_text)
                    sys.stdout.flush()
                    pieces.append(out.delta_text)
                if out.finished:
                    final = out
    except KeyboardInterrupt:
        final = engine.abort_request(rid) or final
        print("  [stopped]")
    e2e = time.perf_counter() - t0
    n = final.num_output_tokens if final else 0
    stats = {
        "tokens": n,
        "prompt_tokens": len(prompt_ids),
        "cached_tokens": final.num_cached_tokens if final else 0,
        "ttft_ms": (ttft or 0) * 1e3,
        "tpot_ms": (e2e - ttft) / (n - 1) * 1e3 if ttft and n > 1 else None,
        "tok_per_s": n / e2e if e2e else 0,
        "finish": final.finish_reason if final else None,
    }
    drafted = engine.stats.spec_draft_tokens - before[0]
    if drafted:
        stats["spec_acceptance"] = (engine.stats.spec_accepted_tokens - before[1]) / drafted
    return "".join(pieces), stats


def format_stats(s: dict) -> str:
    parts = [f"{s['tokens']} tok", f"TTFT {s['ttft_ms']:.0f} ms"]
    if s["tpot_ms"] is not None:
        parts.append(f"TPOT {s['tpot_ms']:.1f} ms")
    parts.append(f"{s['tok_per_s']:.1f} tok/s")
    parts.append(f"prompt {s['prompt_tokens']} tok" + (f" ({s['cached_tokens']} from prefix cache)" if s["cached_tokens"] else ""))
    if "spec_acceptance" in s:
        parts.append(f"draft accepted {s['spec_acceptance']:.0%}")
    if s["finish"]:
        parts.append(f"finish={s['finish']}")
    return "  · ".join(parts)


# ----------------------------------------------------------------------------- subcommands


def cmd_features(args) -> int:
    print(features_table())
    print("\nexamples:\n  tiny-engine chat --features paged,batching,prefix\n"
          "  tiny-engine chat --features all --quantization none\n"
          "  tiny-engine serve --features spec --num-speculative-tokens 6")
    return 0


def cmd_config(args) -> int:
    names = parse_feature_list(args.features)
    if names:
        resolved, _ = resolve_features(names)
        added = [n for n in resolved if n not in names]
        print(f"features: {', '.join(resolved)}" + (f"   (added as required: {', '.join(added)})" if added else ""))
    config = EngineConfig(**config_values(args))
    try:
        config.validate()
    except ValueError as exc:
        print(f"invalid configuration: {exc}")
        return 2
    print_features(config)
    for k, v in asdict(config).items():
        print(f"  {k:<26} {v}")
    return 0


def cmd_generate(args) -> int:
    prompts = (args.prompts or []) + (args.prompt or [])
    if not prompts:
        print("give at least one prompt: tiny-engine generate \"Explain the KV cache\"")
        return 2
    engine = build_engine(args, record_steps=args.show_steps or None)
    for prompt in prompts:
        ids = engine.encode_prompt(prompt) if args.raw else engine.encode_chat(
            ([{"role": "system", "content": args.system}] if args.system else []) + [{"role": "user", "content": prompt}])
        params = engine.sampling_params(**sampling_overrides(args))
        print(f"\n› {prompt}\n")
        engine.step_log.clear()
        _, stats = stream_request(engine, ids, params)
        print(f"\n\n  {format_stats(stats)}")
        if args.show_steps:
            print(f"\n  {'step':>5} {'phase':>8} {'batch':>5} {'tokens in':>9} {'context':>8} {'forward ms':>11} {'sample ms':>10}")
            for i, s in enumerate(engine.step_log):
                print(f"  {i:>5} {s.phase:>8} {s.batch_size:>5} {s.seq_len:>9} {s.context_len:>8} {s.forward_ms:>11.2f} {s.sample_ms:>10.2f}")
    return 0


CHAT_HELP = """commands:
  /reset              forget the conversation
  /system TEXT        set the system message (and reset)
  /set NAME VALUE     temperature | top_p | top_k | max_tokens | seed
  /stats              engine counters (KV usage, prefix hits, speculative acceptance)
  /features           what this engine has switched on
  /exit               quit (also Ctrl-D)"""


def cmd_chat(args) -> int:
    engine = build_engine(args)
    sampling = sampling_overrides(args)
    system = args.system
    history: list[dict] = []
    print("chat ready. Type a message, or /help.  Ctrl-C stops a reply, Ctrl-D quits.")
    while True:
        try:
            line = input("\nyou › ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line.startswith("/"):
            cmd, _, rest = line.partition(" ")
            if cmd in ("/exit", "/quit"):
                return 0
            if cmd == "/help":
                print(CHAT_HELP)
            elif cmd == "/reset":
                history.clear()
                print("conversation cleared")
            elif cmd == "/system":
                system, history = rest or None, []
                print(f"system message {'set' if system else 'cleared'}; conversation cleared")
            elif cmd == "/set":
                name, _, value = rest.partition(" ")
                if name not in ("temperature", "top_p", "top_k", "max_tokens", "seed") or not value:
                    print("usage: /set temperature|top_p|top_k|max_tokens|seed VALUE")
                    continue
                sampling[name] = float(value) if name in ("temperature", "top_p") else int(value)
                print(f"{name} = {sampling[name]}")
            elif cmd == "/stats":
                for k, v in engine.gauges().items():
                    print(f"  {k:<30} {v}")
                if engine.stats.spec_draft_tokens:
                    print(f"  {'spec_acceptance_rate':<30} {engine.stats.spec_acceptance_rate:.2%}")
                print(f"  {'tokens_generated':<30} {engine.stats.generation_tokens}")
            elif cmd == "/features":
                print_features(engine.config)
            else:
                print(f"unknown command {cmd}; /help lists them")
            continue
        history.append({"role": "user", "content": line})
        messages = ([{"role": "system", "content": system}] if system else []) + history
        try:
            ids = engine.encode_chat(messages)
            params = engine.sampling_params(**sampling)
        except ValueError as exc:
            print(f"error: {exc}")
            history.pop()
            continue
        print("\nassistant › ", end="", flush=True)
        text, stats = stream_request(engine, ids, params)
        history.append({"role": "assistant", "content": text})
        print(f"\n  {format_stats(stats)}")


def cmd_serve(args) -> int:
    import uvicorn

    from tiny_engine.serving import build_app

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    engine = build_engine(args)
    print(f"serving on http://{args.host}:{args.port}/v1  (OpenAI-compatible; Ctrl-C to stop)")
    uvicorn.run(build_app(engine), host=args.host, port=args.port, log_level=args.log_level)
    return 0


def cmd_bench(args) -> int:
    if args.stage not in BENCHMARKS:
        print("stages:\n" + "\n".join(f"  {k:<7} {v[2]}" for k, v in BENCHMARKS.items()))
        return 2
    script, preset_args, _ = BENCHMARKS[args.stage]
    path = BENCH_DIR / script
    if not path.exists():
        print(f"benchmark not found: {path} (run from a checkout of 01-tiny-engine, installed with pip install -e .)")
        return 2
    extra = list(args.rest)
    if extra and extra[0] == "--":
        extra = extra[1:]
    if script.endswith(".sh"):
        return subprocess.call(["bash", str(path), *extra])
    sys.argv = [str(path), *preset_args, *extra]
    sys.path.insert(0, str(BENCH_DIR))
    runpy.run_path(str(path), run_name="__main__")
    return 0


# ----------------------------------------------------------------------------- interactive menu


def _ask(prompt: str, default: str = "") -> str:
    try:
        value = input(f"{prompt}" + (f" [{default}]" if default else "") + ": ").strip()
    except EOFError:
        return default
    return value or default


def cmd_menu(args) -> int:
    order = list(FEATURES)
    selected = set(parse_feature_list(args.features))
    model = args.model or EngineConfig.model
    draft = args.speculative_model or DEFAULT_DRAFT_MODEL
    k = args.num_speculative_tokens or 4
    while True:
        print("\n tiny-engine — choose features\n")
        for i, name in enumerate(order, 1):
            f = FEATURES[name]
            mark = "x" if name in selected else " "
            print(f"  [{mark}] {i:>2}  {name:<9} stage {f.stage}  {f.summary}")
        print(f"\n  model: {model}")
        if "spec" in selected:
            print(f"  draft: {draft}  (k = {k})")
        try:
            names, _ = resolve_features(sorted(selected, key=order.index))
            added = [n for n in names if n not in selected]
            if added:
                print(f"  also enabled because required: {', '.join(added)}")
            status = None
        except ValueError as exc:
            status = str(exc)
            print(f"  ! {status}")
        print("\n  numbers toggle features (e.g. 2 6 7) · a = all · n = none · m = model · d = draft · k = draft tokens")
        print("  c = chat · g = generate · s = serve · b = benchmark · v = view config · q = quit")
        try:
            choice = input("\n› ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not choice:
            continue
        tokens = choice.replace(",", " ").split()
        if all(t.isdigit() for t in tokens):
            for t in tokens:
                i = int(t) - 1
                if not 0 <= i < len(order):
                    continue
                name = order[i]
                if name in selected:
                    selected.discard(name)
                else:
                    group = FEATURES[name].group
                    if group:  # alternatives: switching one on switches the others off
                        selected -= {n for n in selected if FEATURES[n].group == group}
                    selected.add(name)
            continue
        cmd = tokens[0]
        if cmd == "q":
            return 0
        if cmd == "a":
            selected = set(ALIASES["all"])
        elif cmd == "n":
            selected = set()
        elif cmd == "m":
            model = _ask("model", model)
        elif cmd == "d":
            draft = _ask("draft model (same tokenizer as the target)", draft)
        elif cmd == "k":
            k = int(_ask("speculative tokens per step", str(k)))
        elif cmd in ("c", "g", "s", "b", "v"):
            if status:
                print(f"  fix this first: {status}")
                continue
            argv = ["--model", model]
            if selected:
                argv += ["--features", ",".join(sorted(selected, key=order.index))]
            if "spec" in selected:
                argv += ["--speculative-model", draft, "--num-speculative-tokens", str(k)]
            if cmd == "c":
                return main(["chat", *argv])
            if cmd == "g":
                prompt = _ask("prompt")
                return main(["generate", *argv, prompt]) if prompt else 0
            if cmd == "s":
                port = _ask("port", "8001")
                return main(["serve", *argv, "--port", port])
            if cmd == "v":
                main(["config", *argv])
                continue
            print("\n" + "\n".join(f"  {k_:<7} {v[2]}" for k_, v in BENCHMARKS.items()))
            stage = _ask("stage", "5")
            extra = shlex.split(_ask("extra benchmark options", ""))
            return main(["bench", stage, *argv, *extra])
        else:
            print(f"  unknown choice {choice!r}")


# ----------------------------------------------------------------------------- entry point


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tiny-engine", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", metavar="COMMAND")

    s = sub.add_parser("menu", help="interactive menu (the default with no command)")
    add_engine_args(s)
    s.set_defaults(func=cmd_menu)

    s = sub.add_parser("features", help="list every feature")
    s.set_defaults(func=cmd_features)

    s = sub.add_parser("config", help="show the resolved configuration without loading a model")
    add_engine_args(s)
    s.set_defaults(func=cmd_config)

    s = sub.add_parser("chat", help="chat in the terminal")
    add_engine_args(s)
    add_sampling_args(s)
    s.add_argument("--system", default=None)
    s.set_defaults(func=cmd_chat)

    s = sub.add_parser("generate", help="run prompts and print text + metrics")
    add_engine_args(s)
    add_sampling_args(s)
    s.add_argument("prompts", nargs="*")
    s.add_argument("--prompt", action="append", default=None, help="another prompt (repeatable)")
    s.add_argument("--system", default=None)
    s.add_argument("--raw", action="store_true", help="no chat template")
    s.add_argument("--show-steps", action="store_true", help="per-step batch size, tokens, latency")
    s.set_defaults(func=cmd_generate)

    s = sub.add_parser("serve", help="OpenAI-compatible HTTP server")
    add_engine_args(s)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8001)
    s.add_argument("--log-level", default="info")
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser("bench", help="run a stage benchmark: " + " | ".join(BENCHMARKS))
    s.add_argument("stage", nargs="?", default="")
    s.add_argument("rest", nargs=argparse.REMAINDER, help="options passed to the benchmark (engine options work too)")
    s.set_defaults(func=cmd_bench)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    if not argv:
        argv = ["menu"]
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    if args.command not in ("serve", "bench"):
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        return args.func(args) or 0
    except ValueError as exc:  # bad feature combination, bad option values
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
