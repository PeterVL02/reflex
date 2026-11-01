# License register

Everything this project uses or ships, and the terms it comes under. Add a row **before** first use, in the same commit.

Intended use: **research only**. This is a public research code repository under Apache-2.0, and only code is published: no data, model weights or results are committed. Any deployment or release of trained weights first needs the `unclear` rows below resolved, in particular WildGuard/JudgeComparison.

Status: `ok` (verified, compatible with intended use) · `unclear` (not verified or open question) · `conflict` (incompatible, must be resolved)

| Component | Type | Version / source | License | Obligations and restrictions | How we use it | Status | Checked |
|---|---|---|---|---|---|---|---|
| neuralchemy/Prompt-injection-dataset (`full`) | dataset | HF, revision `7d70432dfcf47a821612cbf9d34e9d9e3ad20e75` | Apache-2.0 (HF card metadata) | Card names no per-source licences; README lists HackAPrompt CC BY 4.0, HarmBench MIT, WildGuard/JudgeComparison "research", NeurAlchemy Apache-2.0. If rows are ever redistributed, include the dataset licence and attribution for each source | Training and evaluation; downloaded locally, never committed | unclear | 2026-11-01 |
| HackAPrompt (source inside the dataset) | dataset | via neuralchemy dataset | CC BY 4.0 (per README, not re-verified) | Attribution if redistributed | Part of training/eval data | unclear | 2026-11-01 |
| HarmBench (source inside the dataset) | dataset | via neuralchemy dataset | MIT (per README, not re-verified) | Keep the licence notice if redistributed | Part of training/eval data | unclear | 2026-11-01 |
| WildGuard / JudgeComparison (`wildguard_judgecomp`) | dataset | via neuralchemy dataset | "research" (per README; exact terms not verified) | May be research-only. Check before any public or commercial redistribution of rows or of weights trained on them | Part of training/eval data | unclear | 2026-11-01 |
| convaiinnovations/laya | model | HF, sha `7b928d828b7b0e022f929d9bd2e44165aa270148` | Apache-2.0 (HF card metadata) | Fine-tuned checkpoints are derivatives: include the licence and a statement of changes if published | Zero-shot features; fine-tuning base | ok | 2026-11-01 |
| Qwen/Qwen3-Embedding-0.6B | model | HF, sha `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` | Apache-2.0 (HF card metadata) | Keep the licence notice if redistributed | Frozen embeddings | ok | 2026-11-01 |
| laya | library | 0.4.19 (PyPI) | Apache-2.0 | Keep the notice if redistributed | Laya router, fine-tuning | ok | 2026-11-01 |
| datasets | library | 5.1.0 | Apache-2.0 | Keep the notice if redistributed | Dataset download | ok | 2026-11-01 |
| transformers | library | 5.22.1 | Apache-2.0 | Keep the notice if redistributed | Model runtime | ok | 2026-11-01 |
| sentence-transformers | library | 6.2.0 | Apache-2.0 | Keep the notice if redistributed | Embedding runtime | ok | 2026-11-01 |
| torch | framework | 2.11.0+cu128 | BSD-3-Clause | Keep the notice if redistributed | Training and inference | ok | 2026-11-01 |
| scikit-learn | library | 1.9.1 | BSD-3-Clause | Keep the notice if redistributed | Vectorisers, LR/GBM/RF, CV | ok | 2026-11-01 |
| numpy | library | 2.5.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | Keep the notices if redistributed | Arrays | ok | 2026-11-01 |
| pandas | library | 3.1.0 | BSD-3-Clause | Keep the notice if redistributed | Data frames | ok | 2026-11-01 |

Types: model, dataset, framework, library, code, service.

Library licences come from the installed package metadata. Model and dataset licences come from the Hugging Face API card metadata at the listed sha.
