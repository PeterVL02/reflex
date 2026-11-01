# Related work: prompt-injection and jailbreak detection (2024–2026)

Written 2026-11-01 from web research. Each claim links to its source. Unless noted, numbers are the authors' own and were not independently replicated. Where only an abstract was read, the text says so.

## Summary

- **In-distribution scores say little.** When the source dataset is held out (leave-one-dataset-out), detectors lose about 8.4 AUC points compared with same-source splits ([Fomin 2026](https://arxiv.org/html/2602.14161v1)). On agent benchmarks, rankings from one benchmark do not carry over to another: Kendall's τ is about 0.01–0.31 and not significant ([2610.03448](https://arxiv.org/html/2610.03448)). Our 0.997 AUROC is therefore weak evidence of how the model will do in deployment.
- **The field reports TPR at a fixed low FPR, not accuracy.** Recall at 1% FPR is the usual figure ([Prompt Guard 2 card](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M)), with 0.1% and 0.05% used too ([PromptShield](https://arxiv.org/html/2501.15145v2)). AUC is a poor guide to low-FPR behaviour (same source).
- **Over-defence is the main practical failure.** Guards flag benign text that contains words like "ignore". On NotInject, state-of-the-art guards drop to about 60% accuracy ([InjecGuard/PIGuard](https://arxiv.org/abs/2410.22770)). On benign agent tool outputs, ProtectAI v2 has 30–67% FPR ([2610.03448](https://arxiv.org/html/2610.03448)).
- **Every static detector falls to adaptive attacks.** Search attacks get more than 90% ASR against ProtectAI, Prompt Guard and Model Armor, and 71% against PIGuard ([Nasr, Carlini et al. 2025](https://arxiv.org/html/2510.09023)). Emoji or Unicode smuggling reaches up to 100% evasion ([Hackett et al. 2025](https://arxiv.org/html/2504.11168)). A detector is one layer, not the defence.
- **Small fine-tuned encoders are still the practical choice for screening.** Recent examples are Prompt Guard 2 (86M/22M), PIGuard and Sentinel (ModernBERT-large). LLM judges cost a 7–8B forward pass and trailed the best dedicated detectors on agent benchmarks ([2610.03448](https://arxiv.org/html/2610.03448)). A fine-tuned 8B *classifier* does beat DeBERTa clearly at low FPR ([PromptShield](https://arxiv.org/html/2501.15145v2)).
- **Activation probes are the strongest newer idea, and they are cheap.** A logistic-regression probe on Llama-3.1-8B activations reaches 0.912 LODO AUC. On indirect injections it beats Prompt Guard 2 (68.0% vs 37.3% detection) ([Fomin 2026](https://arxiv.org/html/2602.14161v1)). In production, Anthropic runs a linear probe as the first stage of a two-stage cascade ([Constitutional Classifiers++](https://arxiv.org/html/2601.04603v1)).
- **Cascades are standard deployment practice:** a cheap first stage, with an expensive second stage only for flagged or uncertain inputs ([CC++](https://arxiv.org/html/2601.04603v1); [LlamaFirewall](https://arxiv.org/html/2505.03574)).

## 1. Detector models and approaches

### Open guard and classifier models (candidate baselines)

| Model | Size / base | Scope | Notes |
| --- | --- | --- | --- |
| [Llama Prompt Guard 2 86M](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M) | mDeBERTa-base | injection + jailbreak, one binary label | Multilingual (8 languages evaluated), 512-token window. Energy-based loss to cut OOD false positives, tokenizer hardened against whitespace and fragment tricks. Card: AUC .998, recall 97.5% at 1% FPR (English), 92.4 ms on A100 at 512 tokens. Gated, Llama licence. |
| Prompt Guard 2 22M | DeBERTa-xsmall | same | Card: recall 88.7% at 1% FPR, multilingual AUC .942. 19.3 ms on A100 ([LlamaFirewall](https://arxiv.org/html/2505.03574)). |
| [PIGuard](https://huggingface.co/leolee99/PIGuard) (formerly InjecGuard, [ACL 2025](https://aclanthology.org/2025.acl-long.1468)) | DeBERTa-v3-base | injection | Trained with "MOF" to remove trigger-word bias. The authors report +30.8% over the previous best on their benchmarks ([arXiv](https://arxiv.org/abs/2410.22770)). |
| [ProtectAI deberta-v3-base-prompt-injection-v2](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2) | DeBERTa-v3-base | injection only, English | Card: "does not detect jailbreak attacks". Project archived. Weak in independent evaluations (below). |
| [Qualifire Sentinel](https://arxiv.org/abs/2506.05446) | ModernBERT-large | injection / jailbreak | Authors report 0.987 accuracy on their own test set. No independent replication found, except [2610.03448](https://arxiv.org/html/2610.03448), where it is one of 15 detectors. |
| [Qwen3Guard-Gen 0.6B/4B/8B](https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B) ([report](https://arxiv.org/abs/2510.14276)) | Qwen3 | harm categories, including "Jailbreak (only for input)" | 119 languages, labels safe / controversial / unsafe, Apache-2.0. A Stream variant classifies token by token. I found no injection-specific numbers. |
| [Llama Guard 4 12B](https://huggingface.co/meta-llama/Llama-Guard-4-12B) | Llama 4 | harm taxonomy S1–S14 | Does **not** target injection. Its card refers users to Prompt Guard 2 "for detecting prompt attacks". |
| [Granite Guardian 3.3 8B](https://www.ibm.com/granite/docs/models/guardian) ([paper](https://arxiv.org/pdf/2412.07724)) | Granite | configurable criteria, including `jailbreak` | Thinking and non-thinking modes. |
| [gpt-oss-safeguard-20b](https://huggingface.co/openai/gpt-oss-safeguard-20b) | 21B MoE, 3.6B active | judges against a policy you write | Card says it "fits into GPUs with 16GB of VRAM". Apache-2.0. Reasoning effort low/med/high. This is an LLM judge, so it is slow. |

**Independent results disagree with model cards.** PromptShield could not reproduce Prompt Guard's self-reported 71% TPR at 1% FPR on its own benchmark. It measured 12.8% for Prompt Guard 1, 20.4% for InjecGuard and 2.0% for ProtectAI v2 ([PromptShield, Table 4](https://arxiv.org/html/2501.15145v2)). On Lakera's PINT, Prompt Guard 2 86M scores 78.8% and ProtectAI v2 79.1%, against 95.2% for the commercial Lakera Guard ([PINT](https://github.com/lakeraai/pint-benchmark)). Lakera runs PINT itself, so treat that ranking as vendor-run. On agent tool outputs, a 2026 detector (Horizon-Labs, 308M) leads. PIGuard is excellent on BIPIA (95.1% at 1% FPR) but poor on AgentDojo (2.1%) ([2610.03448](https://arxiv.org/html/2610.03448)). **Confidence:** high that per-detector rankings are benchmark-dependent; low on any single headline number.

### Fine-tuned small encoders vs LLM-as-judge

- Encoders win on cost, and on agent benchmarks they also win on accuracy. Qwen2.5-7B and Llama-3.1-8B judges stayed below the best dedicated detectors, and Llama-3.1-8B reached only 8.3% on BIPIA at 1% FPR ([2610.03448](https://arxiv.org/html/2610.03448)). Llama-as-judge detected 7.1% of indirect injections ([Fomin 2026](https://arxiv.org/html/2602.14161v1)).
- Frontier judges do better, but they cost money. PromptArmor with GPT-4o/4.1/o4-mini reports FPR and FNR both below 1% on AgentDojo ([2507.15219](https://arxiv.org/abs/2507.15219), abstract only). That rules them out for us (no paid APIs).
- A fine-tuned LLM used as a classifier is a different thing from a judge. A LoRA-tuned Llama-3.1-8B reached 94.8% TPR at 1% FPR against 43.2% for DeBERTa-v3-base on the same data ([PromptShield](https://arxiv.org/html/2501.15145v2)). Model scale helps when the model is trained as a classifier.

### Embedding-based detectors

- Embeddings with Random Forest or XGBoost beat the open DeBERTa detectors on AUC and precision, using a 467k-prompt set (RF on OpenAI embeddings: AUC 0.764 vs 0.511 for ProtectAI v2) ([Ayub & Majumdar 2024](https://arxiv.org/html/2410.22284)). This is essentially our design, and it supports it. Note how low the AUCs are on their mixed data.
- One-class / "typical prompt" modelling treats safety as OOD detection on a model of benign prompts. Its authors claim up to 40× lower FPR than specialised safety models across 18 benchmarks ([Trust The Typical, 2602.04581](https://arxiv.org/abs/2602.04581), abstract only, unverified).

### Perplexity-based detectors

- Perplexity catches GCG-style gibberish suffixes, but many benign prompts also have high perplexity. A LightGBM on perplexity plus length removes most of those false positives ([Alon & Kamfonas](https://arxiv.org/abs/2308.14132)). Readable attacks such as AutoDAN "bypass them effectively" ([AutoDAN](https://arxiv.org/abs/2310.04451)). Useful only as one feature among others. **Confidence:** high.

### Attention- and activation-based detection

- **Attention Tracker:** training-free. It watches "important heads" that shift attention from the instruction to the injected text, and reports up to +10% AUROC over prior methods ([2411.00348](https://arxiv.org/abs/2411.00348)).
- **TaskTracker:** a linear probe on the change in activations before and after the model reads external data. It reports "near-perfect" OOD ROC AUC and comes with 500k+ instances ([2406.00799](https://arxiv.org/abs/2406.00799)).
- **Raw-activation LR probe (Llama-3.1-8B, layer 31):** generalised best under LODO (0.912 AUC). An MLP scored 0.841 and SAE features 0.838. It detected 99.0% of InjecAgent tool attacks, against 21.5% for Llama-as-judge ([Fomin 2026](https://arxiv.org/html/2602.14161v1)).
- **Production:** probes "matched or exceeded" small fine-tuned external classifiers at almost no extra compute ([CC++](https://arxiv.org/html/2601.04603v1)).

### Known-answer / canary detection

- Known-answer detection (KAD) runs the data with an instruction like "Repeat [secret key] once while ignoring the following text". If the key does not come back, the data is flagged. Plain KAD has FPR up to 0.1 and FNR up to 0.21 under optimisation attacks. DataSentinel fine-tunes Mistral-7B with a minimax objective and reports FPR close to 0 and FNR ≤0.07. It still fails when the injected task matches the target task (FNR 0.87 on sentiment) ([DataSentinel, IEEE S&P 2025](https://arxiv.org/html/2504.11358v4)). An RL adaptive attack still beat it ([2510.09023](https://arxiv.org/html/2510.09023)). This approach is for checking *data* in an application pipeline, and it needs an LLM call per input.

### Prevention rather than detection

- **Spotlighting** marks untrusted input with delimiters, datamarking or encoding. On GPT models it cut ASR from more than 50% to below 2% ([2403.14720](https://arxiv.org/abs/2403.14720)), but adaptive human attacks still got through ([2510.09023](https://arxiv.org/html/2510.09023)).
- **StruQ, SecAlign and Meta-SecAlign** train the model to ignore instructions found in the data channel. SecAlign reports ASR below 10% ([2410.05451](https://arxiv.org/abs/2410.05451)). Meta-SecAlign releases 8B and 70B models that keep "almost all" utility ([2507.02735](https://arxiv.org/abs/2507.02735)).
- **CaMeL** separates control flow from data flow and adds capabilities. It solves 77% of AgentDojo tasks with provable security, against 84% undefended ([2503.18813](https://arxiv.org/abs/2503.18813)).
- **Relevance to us:** these sit in the downstream application, not in a screening layer. Recommend them to the organisation as complements, since a screen alone will be bypassed.

## 2. Benchmarks and datasets

**First check for overlap with our training data.** Our data already contains HackAPrompt, HarmBench and WildGuard rows (README), so those are *not* OOD. WildJailbreak and WildGuardMix come from the same AI2 pipeline. Deduplicate any new set against our training texts (exact match plus embedding near-duplicates) before scoring.

| Dataset | What it is | OOD value for us |
| --- | --- | --- |
| [NotInject](https://huggingface.co/datasets/leolee99/NotInject) | 339 benign prompts containing 1–3 injection trigger words, MIT | **High.** A direct over-defence test. |
| [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) | 662 rows, Apache-2.0, many German-politics topics | **High.** Small and cheap. Check for a German-language share. |
| [Lakera gandalf_ignore_instructions](https://huggingface.co/datasets/Lakera/gandalf_ignore_instructions) | 1,000 real Gandalf game injections, MIT | Medium-high. Positives only, so it measures TPR. |
| [Qualifire / rogue-security benchmark](https://huggingface.co/datasets/qualifire/prompt-injections-benchmark) | 5,000 jailbreak/benign, gated, CC BY-NC | High. Fine for internal evaluation. Used as an OOD regime in [2605.26999](https://arxiv.org/abs/2605.26999). |
| [JBB-Behaviors](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) ([paper](https://arxiv.org/abs/2404.01318)) | 100 harmful + 100 benign behaviours, MIT, plus attack artifacts | Medium. Partly overlaps HarmBench sources. The benign half is useful. |
| [XSTest](https://arxiv.org/abs/2308.01263) | 250 safe prompts that look unsafe, 200 unsafe contrast prompts | High for over-defence on the harm side. |
| [OR-Bench](https://arxiv.org/abs/2405.20947) | 80k over-refusal prompts, ~1k hard, 600 toxic | **High.** Use hard-1K as benign hard negatives. |
| [WildJailbreak](https://arxiv.org/abs/2406.18510) | 262k vanilla and adversarial pairs, including benign prompts that look harmful | Medium. Same family as WildGuard, so dedupe first. |
| ["In-the-wild" jailbreaks](https://arxiv.org/abs/2308.03825) | 1,405 jailbreak prompts, Dec 2022 – Dec 2023 | Medium. Real-world role-play and DAN-style prompts. |
| [BIPIA](https://github.com/microsoft/BIPIA) ([paper](https://arxiv.org/abs/2312.14197)) | Indirect injection in email, web, table, summarisation and code contexts | High if documents are screened. Some contexts must be rebuilt for licence reasons. Repo archived. |
| [LLMail-Inject](https://huggingface.co/datasets/microsoft/llmail-inject-challenge) ([paper](https://arxiv.org/abs/2506.09956)) | 208,095 human-written adaptive attacks on an email agent | **High.** Real adaptive attackers. Pair it with benign emails. |
| [AgentDojo](https://arxiv.org/abs/2406.13352) | 97 tasks, 629 security cases, tool outputs | High for agent use. Regenerate the matched/shift splits with [this code](https://github.com/lzwhehe/benign-instruction-bench). |
| [PINT](https://github.com/lakeraai/pint-benchmark) | 4,314 inputs, 30% non-English, 20.9% hard negatives | Very high, but mostly proprietary: only an example set is public. Running it on your own detector means contacting Lakera. |
| [GenTel-Bench](https://arxiv.org/abs/2409.19521) | 84,812 attacks, 28 scenarios | Medium. Large; benign composition not checked. |
| [PromptShield](https://arxiv.org/html/2501.15145v2) | Benign conversational (LMSYS, UltraChat) and application-structured data | Medium. Its training split includes HackAPrompt. |

A ready-made LODO framework over 18 datasets exists: [prompt-mining](https://github.com/maxf-zn/prompt-mining) ([paper](https://arxiv.org/html/2602.14161v1)).

## 3. Known failure modes

- **Over-defence:** caused by trigger-word bias ([NotInject](https://arxiv.org/abs/2410.22770)). Structured inputs such as markup and YAML drive false positives, and removing the markup lowers FPR at the default threshold ([2610.03448](https://arxiv.org/html/2610.03448)). The 20.9% hard-negative share in PINT exists for this reason ([PINT](https://github.com/lakeraai/pint-benchmark)). ProtectAI's card advises against scanning system prompts because of false positives ([card](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2)).
- **Obfuscation:** character injection across six guardrails. Emoji smuggling reached 100% evasion for both injection and jailbreak, Unicode tag smuggling 90%/82%, and bidirectional text 79%/99%. Word-importance rankings from white-box models transfer to black-box guards ([Hackett et al.](https://arxiv.org/html/2504.11168)). Almost all of this is fixable by normalisation before the classifier.
- **Adaptive attacks:** 12 defences broken, most above 90% ASR, and human red-teaming succeeded in every case evaluated ([2510.09023](https://arxiv.org/html/2510.09023)). Eight indirect-injection defences were broken with more than 50% ASR ([2503.00061](https://arxiv.org/abs/2503.00061)). The cards say the same: "adversaries may develop sophisticated attacks specifically to bypass detection" ([Prompt Guard 2](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M)).
- **Multilingual:** guards trained mostly on English degrade. Translate-then-detect and multilingual fine-tuning help without closing the gap. Examples: SEALGuard reports a +48% gain over LlamaGuard on SEA languages ([2507.08898](https://arxiv.org/pdf/2507.08898v1)), and semantic-only cross-lingual detection collapses to under 7% recall on heterogeneous data ([2604.25716](https://www.emergentmind.com/papers/2604.25716)). **Confidence:** medium. These come from search snippets of single papers.
- **Indirect injection:** detectors trained on user prompts miss injections in tool outputs. Prompt Guard 2 caught 37.3% of indirect injections ([Fomin](https://arxiv.org/html/2602.14161v1)). Task-irrelevant injections are missed by almost every detector except PIGuard ([2610.03448](https://arxiv.org/html/2610.03448)). Hidden PDF injections do not generalise across families ([CrackedPDFs](https://arxiv.org/html/2607.19396v1)).
- **Distribution shift and source shortcuts:** a classifier can tell which dataset a prompt came from with 96.6% accuracy, and 28% of the top SAE features are dataset shortcuts ([Fomin](https://arxiv.org/html/2602.14161v1)). Our README already notes that severity follows the source. The binary label may do the same.
- **Scope mismatch:** our label merges injection, jailbreak and harmful requests (HarmBench). Prompt Guard 2 and PIGuard target attacks, not harmful content, and Llama Guard targets harm, not injection ([LG4 card](https://huggingface.co/meta-llama/Llama-Guard-4-12B)). Baseline comparisons must therefore be broken down by subset.

## 4. Evaluation practice

- **Metrics:**
  - TPR (recall) at 1% FPR ([Prompt Guard 2](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M); [2610.03448](https://arxiv.org/html/2610.03448)), and at 0.5%, 0.1% and 0.05% FPR ([PromptShield](https://arxiv.org/html/2501.15145v2)).
  - FPR on clean benign sets, and the share of tasks blocked ([2610.03448](https://arxiv.org/html/2610.03448)).
  - ASR together with utility under attack ([LlamaFirewall](https://arxiv.org/html/2505.03574)). Prompt Guard 2 also reports "APR at 3% utility reduction" on AgentDojo ([card](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M)).
  - Over-defence accuracy on NotInject ([PIGuard](https://arxiv.org/abs/2410.22770)).
- **Robustness protocols:**
  - Leave-one-dataset-out ([Fomin](https://arxiv.org/html/2602.14161v1)).
  - Matched vs shifted splits, and a check for training-set contamination ([2610.03448](https://arxiv.org/html/2610.03448)).
  - Character perturbation suites ([Hackett](https://arxiv.org/html/2504.11168)).
  - Adaptive attacks: gradient, RL, search and human ([2510.09023](https://arxiv.org/html/2510.09023)).
  - Repeated splits plus thresholded deployment metrics, because "ranking performance does not reliably predict deployment effectiveness" ([2605.26999](https://arxiv.org/abs/2605.26999), abstract only).
- **Thresholds:** PromptShield calibrates its thresholds on the evaluation data. When scoring OOD, fix the threshold on in-distribution validation data and *then* report FPR and TPR OOD, which is the deployment-realistic way.

## 5. Deployment practice

- **Latency:**
  - Encoder guards: Prompt Guard 2 takes 19 ms (22M) to 92 ms (86M) per 512-token call on an A100 ([LlamaFirewall](https://arxiv.org/html/2505.03574)).
  - LLM checks: example AlignmentCheck calls in LlamaFirewall's trace took 0.86–1.5 s each, and the paper calls them "significantly higher latency" (same source).
  - No paper I read states an explicit latency budget. **Confidence:** low on what a "standard" budget would be.
- **Cascades:**
  - Anthropic: linear probe, then an ensemble. About 5.5% of traffic goes to a fine-tuned Haiku second stage. The flag rate fell from 0.38% to 0.05%, compute overhead dropped about 40× against the earlier exchange classifier, and relative overhead is about 3.5% ([CC++](https://arxiv.org/html/2601.04603v1)).
  - LlamaFirewall: "PromptGuard provides early, low-cost filtering", with AlignmentCheck behind it. On AgentDojo, ASR is 17.6% with no guardrail, 7.5% with PG2, 2.9% with AlignmentCheck and 1.75% combined, while utility goes from 47.7% to 42.7% ([LlamaFirewall](https://arxiv.org/html/2505.03574)).
- **Long inputs:** split them into 512-token windows and scan the windows in parallel ([Prompt Guard 2 card](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M)). Laya's roughly 320-token window will need the same treatment.
- **Calibration and thresholds:**
  - Cards recommend fine-tuning or re-thresholding on your own application data ([Prompt Guard 2](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M)).
  - Measure FPR on your own traffic ([2610.03448](https://arxiv.org/html/2610.03448)).
  - Set the threshold for a target FPR, because the default of 0.5 can be far from it: Prompt Guard at 0.5 gave 22.8% TPR at 2.9% FPR ([PromptShield](https://arxiv.org/html/2501.15145v2)).
- **Defence in depth:** combine the screen with spotlighting, instruction-hierarchy or SecAlign models, and least-privilege tools, since any detector is eventually bypassed ([2510.09023](https://arxiv.org/html/2510.09023); [CaMeL](https://arxiv.org/abs/2503.18813)).

## Implications for this project

Ranked by expected value per unit of effort. GPU times are rough estimates for an RTX 5080, not measurements. Anything likely to need more than about 1 h of GPU is marked **needs user approval**.

1. **Build an OOD evaluation harness, and report TPR at 1% FPR plus FPR on benign sets.**
   - *Value:* very high. It is the open question, and it turns "96.7%" into a number that means something.
   - *Datasets:* NotInject, deepset/prompt-injections, Lakera gandalf_ignore_instructions, the Qualifire benchmark (gated), JBB-Behaviors, XSTest, OR-Bench-hard-1K, and a 5k sample of the in-the-wild jailbreaks.
   - *Steps:* dedupe against training (exact match plus Qwen cosine above about 0.95). Fix the threshold on our validation data. Report per dataset and per scope (injection / jailbreak / harmful / benign).
   - *Effort:* about 1 day. *Compute:* features for about 10k prompts with our cached pipeline, roughly minutes of GPU.
2. **Leave-one-source-out CV on neuralchemy's own sources** (hackaprompt, harmbench, wildguard, neuralchemy_v1, original).
   - *Value:* high. It shows whether the GBM learned source shortcuts, which is [Fomin](https://arxiv.org/html/2602.14161v1)'s main finding.
   - *Effort:* a few hours, reusing `make_splits`. *Compute:* CPU only, since the features are cached.
3. **Benchmark open guards as baselines on the same harness:** Prompt Guard 2 86M and 22M (gated, accept the Llama licence), PIGuard, ProtectAI v2 (as a weak reference), Qualifire Sentinel and Qwen3Guard-Gen-0.6B.
   - *Value:* high. We need to know whether we beat off-the-shelf models outside our own data.
   - *Effort:* about half a day. *Compute:* under 1 h of GPU in total for six models of 0.1–0.8B over about 15k prompts.
4. **Unicode normalisation and an obfuscation test suite.**
   - *Normalise:* apply NFKC, and strip Unicode tag characters (U+E0000–E007F), zero-width characters, variation selectors and bidi controls before every model. Count stripped characters as a handcrafted feature.
   - *Test:* re-score the test split under emoji smuggling, homoglyph, leetspeak, spacing and base64 perturbations ([Hackett](https://arxiv.org/html/2504.11168)).
   - *Value:* high. It is cheap and closes 100%-evasion holes.
   - *Effort:* about half a day. *Compute:* minutes.
5. **Hard-negative augmentation and FPR-targeted thresholds.**
   - Add benign trigger-word text to training, in the spirit of PIGuard's MOF ([2410.22770](https://arxiv.org/abs/2410.22770)). Sources: OR-Bench (not the hard-1K used for evaluation), XSTest-style safe prompts, and benign security discussion and documentation.
   - Pick the operating threshold for a target FPR of 1%, not 0.5.
   - *Value:* high for false positives on real users.
   - *Effort:* 1 day. *Compute:* CPU plus minutes of GPU for features. Re-fine-tuning Laya with it would likely exceed 1 h, so that part **needs user approval**.
6. **Build a two-stage cascade and measure latency.**
   - *Stage 1:* TF-IDF + LR + handcrafted features. Stop when the score is confidently low or high.
   - *Stage 2:* escalate the uncertain band to Qwen + Laya + GBM.
   - *Report:* the share escalated, p50/p95 latency, and TPR at 1% FPR against the full model ([CC++](https://arxiv.org/html/2601.04603v1) pattern).
   - *Value:* medium-high for deployment. *Effort:* 1 day. *Compute:* minutes.
7. **Activation-probe feature.**
   - Train an LR probe on mean-pooled middle- and late-layer hidden states of a small decoder (Qwen3-0.6B or 1.7B; Qwen3-Embedding's own hidden layers are an option), and add it as a combiner input. Evaluate under LODO.
   - *Value:* medium-high. This is the best-generalising approach in [Fomin](https://arxiv.org/html/2602.14161v1), although that result used an 8B model.
   - *Effort:* 1–2 days. *Compute:* feature extraction over 16k plus OOD prompts at about 1.7B should take well under 1 h. An 8B probe would likely be more than 1 h (**needs user approval**).
8. **Indirect-injection and agent evaluation**, only if documents or tool outputs will pass through the screen: LLMail-Inject (a sample), BIPIA and the AgentDojo regenerated splits ([code](https://github.com/lzwhehe/benign-instruction-bench)). Chunk long inputs into windows.
   - *Value:* medium, depending on scope. *Effort:* 1–2 days. *Compute:* under 1 h for a 20k-sample subset. Regenerating AgentDojo traces needs an agent LLM, which would likely be more than 1 h (**needs user approval**).
9. **Multilingual check.**
   - Use PINT-style languages. Evaluate the multilingual Laya checkpoint and Prompt Guard 2 86M on machine-translated copies of the OOD sets.
   - *Value:* medium, depending on whether the organisation's users write in other languages. Translation with a local MT model over about 10k prompts would likely be more than 1 h (**needs user approval**).
10. **LLM-judge ceiling (optional):** gpt-oss-safeguard-20b (fits in 16 GB) or Granite Guardian 3.3 8B on the uncertain band only.
    - *Value:* low-medium. It shows how much a stage-2 judge would add.
    - *Compute:* reasoning-mode judging over thousands of prompts would likely be more than 1 h (**needs user approval**). Evidence that 7–8B judges help is weak ([2610.03448](https://arxiv.org/html/2610.03448)).

**Not recommended now:** building our own adaptive-attack red team (expensive, and the literature already shows static detectors fall); KAD/DataSentinel (it targets application data, costs an LLM call per input, and has itself been broken); perplexity filters as a stand-alone detector.

**Message for the organisation:** a screening layer reduces risk but cannot be the only control. Pair it with spotlighting or structured prompts and least-privilege tools in the application.
