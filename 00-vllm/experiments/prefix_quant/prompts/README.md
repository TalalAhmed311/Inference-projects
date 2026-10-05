# Prompt banks

Generated files are not checked in (large). Rebuild before experiments:

```bash
source ../../.venv/bin/activate   # or /mnt/data/inference/00-vllm/.venv
python ../build_prompts.py
```

Creates `prompts_<size>.jsonl`, `prompts_noprefix_<size>.jsonl`, `shared_prefix.txt`, `manifest.json`.
