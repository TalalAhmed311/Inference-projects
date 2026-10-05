#!/usr/bin/env python3
"""Build fixed prompt banks WITH and WITHOUT a shared prefix.

WITH prefix (prompts_<size>.jsonl):
  Every prompt starts with the same SHARED_PREFIX so vLLM prefix caching can hit.

WITHOUT prefix (prompts_noprefix_<size>.jsonl):
  Each prompt starts with a unique document id + unique body so prefix caching
  cannot usefully reuse earlier KV blocks (same idea as the Stage-0 random bench).

Also writes shared_prefix.txt and manifest.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer

OUT = Path(__file__).resolve().parent / "prompts"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
# Match original Report.md input lens where possible; 1024 is the extra mid point.
TARGET_SIZES = (128, 512, 1024, 2048)
PROMPTS_PER_SIZE = 256  # supports original num_requests = max(8, 4*conc) at conc=64
SHARED_TARGET_TOKENS = 64  # short enough to leave room for size=128 prompts

POLICY_PARAS = [
    "You are a careful enterprise knowledge assistant for Acme Corp. Answer only from the "
    "provided policy excerpt. If the excerpt does not contain the answer, say you do not know. "
    "Prefer short, precise answers. Do not invent policy numbers or dates.",
    "Acme Corp Remote Work Policy (excerpt). Employees may work remotely up to three days per "
    "week after manager approval. Core collaboration hours are 10:00-15:00 in the employee's "
    "local timezone. Laptop disk encryption and MFA are mandatory. VPN is required for access "
    "to production systems. Personal devices may not store customer PII.",
    "Expense Policy. Business travel under $75 does not require pre-approval. Meals are capped "
    "at $60/day domestically and $90/day internationally. Alcohol is not reimbursable. "
    "Receipts must be uploaded within 14 days. Managers must approve expenses above $500 "
    "within five business days.",
    "Incident Response. Severity-1 outages require paging the on-call within 5 minutes. "
    "Customer-data exposure must be reported to Security within 1 hour. Change freezes apply "
    "during major retail events. Rollback plans are required for production schema changes.",
    "Support SLAs. Priority-1 tickets acknowledge in 15 minutes and resolve or mitigate in "
    "4 hours. Priority-2 acknowledge in 1 hour and resolve in 1 business day. Priority-3 "
    "acknowledge in 1 business day. Escalations go Tier1 → Tier2 → Engineering on-call.",
]

UNIQUE_QUESTIONS = [
    "How many remote days per week are allowed after approval?",
    "Are personal devices allowed to store customer PII?",
    "What is the domestic meal cap?",
    "When must receipts be uploaded?",
    "Who must approve expenses above $500?",
    "How quickly must a Severity-1 outage page on-call?",
    "What is the Priority-1 acknowledgment SLA?",
    "Is alcohol reimbursable under the expense policy?",
    "Is VPN required for production system access?",
    "What are the core collaboration hours?",
    "What is required before a production schema change?",
    "Where do ticket escalations go after Tier2?",
    "What is the international meal cap?",
    "Must MFA be enabled on laptops?",
    "When do change freezes apply?",
    "What is the Priority-2 resolve target?",
    "Does travel under $75 need pre-approval?",
    "How soon must customer-data exposure be reported?",
    "What encryption requirement applies to laptops?",
    "What is the Priority-3 acknowledgment target?",
    "Can employees work remotely without manager approval?",
    "What is the manager approval window for large expenses?",
    "Is disk encryption optional?",
    "What should you answer if the excerpt lacks the fact?",
    "Summarize the remote-work eligibility rule in one sentence.",
    "List the mandatory security controls for remote workers.",
    "Give the Priority-1 resolve-or-mitigate deadline.",
    "State whether alcohol expenses can be submitted.",
    "Name the first escalation hop after Tier1.",
    "What timezone are core hours measured in?",
    "Is a rollback plan optional for schema changes?",
    "Quote the rule about personal devices and PII.",
]

FILLER = (
    " Context note: apply this policy consistently across offices, contractors, and "
    "temporary staff unless a written exception exists. "
)


def fit_tokens(tok, text: str, target: int) -> str:
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) >= target:
        return tok.decode(ids[:target])
    pad = FILLER
    while len(tok.encode(text + pad, add_special_tokens=False)) < target:
        text += pad
    ids = tok.encode(text, add_special_tokens=False)[:target]
    return tok.decode(ids)


def fit_shared(tok, target_tokens: int) -> str:
    return fit_tokens(tok, "\n\n".join(POLICY_PARAS), target_tokens)


def fit_prompt_with_shared(tok, shared: str, question: str, target_tokens: int) -> str:
    q = f"\n\nQuestion: {question}\nAnswer briefly using only the policy excerpt."
    body = shared
    while True:
        candidate = body + FILLER + q
        if len(tok.encode(candidate, add_special_tokens=False)) >= target_tokens:
            break
        body += FILLER
    ids = tok.encode(body + q, add_special_tokens=False)
    if len(ids) > target_tokens:
        q_ids = tok.encode(q, add_special_tokens=False)
        shared_ids = tok.encode(shared, add_special_tokens=False)
        keep_mid = max(0, target_tokens - len(shared_ids) - len(q_ids))
        mid = ids[len(shared_ids) : len(shared_ids) + keep_mid]
        ids = (shared_ids + mid + q_ids)[:target_tokens]
    return tok.decode(ids)


def fit_prompt_unique(tok, idx: int, question: str, target_tokens: int) -> str:
    """Unique head so no two prompts share a long prefix."""
    unique_head = (
        f"[DOC-{idx:05d}-{idx * 7919:08x}] Independent case file for request {idx}. "
        f"This document is intentionally unique and must not share a prefix with other cases. "
        f"Seed-paragraph: office_{(idx % 17)} region_{(idx % 9)} revision_{(idx % 101)}. "
    )
    # Shuffle policy paragraph order by idx so bodies diverge early.
    paras = POLICY_PARAS[idx % len(POLICY_PARAS) :] + POLICY_PARAS[: idx % len(POLICY_PARAS)]
    body = unique_head + "\n\n".join(paras)
    q = f"\n\nQuestion: {question} (case_id={idx:03d})\nAnswer briefly using only this case file."
    while len(tok.encode(body + FILLER + q, add_special_tokens=False)) < target_tokens:
        body += f" Extra unique note {idx}-{len(body)}: {FILLER}"
    ids = tok.encode(body + q, add_special_tokens=False)
    if len(ids) > target_tokens:
        q_ids = tok.encode(q, add_special_tokens=False)
        head_ids = tok.encode(unique_head, add_special_tokens=False)
        keep_mid = max(0, target_tokens - len(head_ids) - len(q_ids))
        mid = ids[len(head_ids) : len(head_ids) + keep_mid]
        ids = (head_ids + mid + q_ids)[:target_tokens]
    return tok.decode(ids)


def write_bank(path: Path, rows: list[dict]) -> dict:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    actuals = [r["actual_tokens"] for r in rows]
    return {
        "file": path.name,
        "count": len(rows),
        "token_min": min(actuals),
        "token_max": max(actuals),
        "token_mean": round(sum(actuals) / len(actuals), 1),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    shared = fit_shared(tok, SHARED_TARGET_TOKENS)
    shared_tokens = len(tok.encode(shared, add_special_tokens=False))
    (OUT / "shared_prefix.txt").write_text(shared)

    manifest = {
        "model_tokenizer": MODEL,
        "shared_target_tokens": SHARED_TARGET_TOKENS,
        "shared_actual_tokens": shared_tokens,
        "prompts_per_size": PROMPTS_PER_SIZE,
        "sizes_prefix": {},
        "sizes_noprefix": {},
    }

    for size in TARGET_SIZES:
        if size <= shared_tokens + 24:
            raise SystemExit(f"target size {size} too small for shared prefix ({shared_tokens})")

        prefix_rows = []
        noprefix_rows = []
        for i in range(PROMPTS_PER_SIZE):
            question = UNIQUE_QUESTIONS[i % len(UNIQUE_QUESTIONS)]
            q = f"{question} (case_id={i:03d})"

            text_p = fit_prompt_with_shared(tok, shared, q, size)
            n_p = len(tok.encode(text_p, add_special_tokens=False))
            prefix_rows.append(
                {
                    "id": f"prefix_size{size}_{i:03d}",
                    "mode": "prefix",
                    "target_tokens": size,
                    "actual_tokens": n_p,
                    "shared_tokens": shared_tokens,
                    "unique_question": q,
                    "prompt": text_p,
                }
            )

            text_n = fit_prompt_unique(tok, i, question, size)
            n_n = len(tok.encode(text_n, add_special_tokens=False))
            noprefix_rows.append(
                {
                    "id": f"noprefix_size{size}_{i:03d}",
                    "mode": "noprefix",
                    "target_tokens": size,
                    "actual_tokens": n_n,
                    "shared_tokens": 0,
                    "unique_question": q,
                    "prompt": text_n,
                }
            )

        meta_p = write_bank(OUT / f"prompts_{size}.jsonl", prefix_rows)
        meta_n = write_bank(OUT / f"prompts_noprefix_{size}.jsonl", noprefix_rows)
        manifest["sizes_prefix"][str(size)] = meta_p
        manifest["sizes_noprefix"][str(size)] = meta_n
        # backward-compat key
        manifest.setdefault("sizes", {})[str(size)] = meta_p
        print(f"size={size}: prefix~{meta_p['token_mean']} tok, noprefix~{meta_n['token_mean']} tok, n={PROMPTS_PER_SIZE}")

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
