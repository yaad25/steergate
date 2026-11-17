# SteerGate (Master Plan): Validate First, Build Second

**One-line summary:** A fail-closed empirical validation harness and replication study that tests whether inference-time activation steering measurably improves mathematical reasoning on small local language models—without weight updates or retraining—once sampling variance, prompt baselines, teacher leakage, and hyperparameter snooping are strictly controlled.

> **Why This Plan Cannot Lose (The Zero-Hype Scientific Fork):**
> Most open-source steering projects fail because they build a complex CLI wrapper first and assume activation steering yields $+5\%$ to $+10\%$ gains. When real-world testing shows the vector barely beats a 1-line prompt, the tool becomes useless.
>
> **SteerGate inverts the build order:** We spend Days 1–10 running a 300-line statistically powered evaluator through three cheap **Kill Gates** on free Kaggle GPUs.
> * **If the vector passes (`GO`):** We ship the first verified activation-steering harness proven to beat both Chain-of-Thought (CoT) prompting and random-direction controls on the full 1,319-question GSM8K benchmark.
> * **If the vector fails (`NO-GO`):** We publish a high-impact replication study proving that published inference-time math steering recipes are statistically indistinguishable from a 1-sentence dense prompt or equal-norm random perturbation. **Both outcomes produce a citable, high-value artifact.**

---

## 1. THE EMPIRICAL REALITY: WHY NAIVE STEERING PLANS FAIL

Recent mechanistic interpretability literature exposes four hard constraints that any credible project must address on Day 1:

1. **Published Math Lifts Are $\approx +2.0$ Points, Not $+5.0$ Points:**
   In **DenseSteer** (ICML 2026), steering `Qwen2.5-1.5B-Instruct` and `Qwen2.5-3B-Instruct` from verbose "sparse reasoning" toward compressed "dense reasoning" improved GSM8K accuracy by **$+2.0$ to $+2.1$ percentage points** (e.g., $63.1\% \rightarrow 65.1\%$ on `Qwen2.5-1.5B-Instruct`, and $83.8\% \rightarrow 85.9\%$ on `Qwen2.5-3B-Instruct`). Furthermore, simple random paraphrasing captured nearly a third of that gain.
2. **Small-Sample Evaluations Are Blind to a $+2.0$ Point Signal:**
   On a 100-question evaluation slice around baseline accuracy $p = 0.63$, the binomial standard error is:
   $$\text{SE}_{100} = \sqrt{\frac{0.63 \times 0.37}{100}} \approx 4.83\text{ percentage points}$$
   This creates a 95% confidence interval of $\pm 9.46$ percentage points (nearly $19$ points wide). A 100-question test cannot detect a real $+2.0$ point improvement, while a lucky random seed easily fakes a $+5.0$ point mirage.
3. **The AxBench Boundary (Simple Prompts Frequently Beat Steering Vectors):**
   Stanford's **AxBench** (Wu et al., 2025) demonstrated that across concept-steering benchmarks, a 1-sentence natural language instruction consistently matches or beats activation addition, while increasing the steering multiplier $\alpha$ monotonically degrades generation fluency ("oversteering"). Testing a steering vector without a matched **Prompt-Only Dense Baseline** is scientifically invalid.
4. **The Teacher-Leakage Confounder:**
   When positive steering traces ($y^+$) are rewritten by a frontier model (such as GPT-5.1 in DenseSteer), the extracted vector mixes *step-density* with distilled teacher knowledge. A rigorous evaluation must test whether purely **on-policy self-compressed traces** (requiring zero external API keys) achieve the same lift.

---

## 2. VULNERABILITY PATCH MATRIX (WHAT MAKES THIS VERSION AIRTIGHT)

Every methodological, statistical, and engineering flaw identified across previous reviews is closed before a single GPU hour is spent:

| Vulnerability in Earlier Versions | Why It Breaks Execution | The SteerGate Master Fix |
| :--- | :--- | :--- |
| **1. Test-Set Hyperparameter Snooping on $(L, \alpha)$** | Sweeping 5 layers $\times$ 4 multipliers on the test set guarantees a fake $+1.5\%$ winner by multiple-hypothesis chance ($p$-hacking). | **Strict Train-Split Lock (Gate 1.5):** Vector extraction uses `GSM8K-train[0:150]`. Layer $L^*$ and multiplier $\alpha^*$ are selected strictly on `GSM8K-train[150:350]` ($N = 200$) + $50$ WikiText-2 passages and locked via SHA-256 in `locked_config.json` before `GSM8K-test` is ever loaded. |
| **2. HuggingFace Batch Seeding Drift** | Calling `torch.manual_seed()` before `model.generate()` fails in batches: as soon as Question #1 generates 1 extra token in the steered arm, the CUDA PRNG state shifts for every other question in the batch, destroying paired variance reduction. | **Step-Indexed Gumbel-Max Common Random Numbers (CRN):** Every sampling step $(q, k, t)$ uses a deterministic counter hash `seed(master_seed, qid, k, token_step)` to draw uniform noise independent of batch size or sequence length divergence. |
| **3. The Gate 2 Compute Wall ($52,760+$ Rollouts)** | Running $1,319 \times 8 \times 6\text{ arms} = 63,312$ rollouts unbatched takes $>30$ hours and dies when Kaggle sessions time out at 12 hours. | **Dual-T4 Sharding + Append-Only JSONL Checkpointing + Two-Stage G2:** Shards questions across both GPUs on Kaggle's free `T4 x2` instance (`batch_size = 32`, `sdpa`), flushes every completed question to `.jsonl` for 1-second crash recovery, and checks the noise floor (`G2-a`) before the full sweep (`G2-b`). |
| **4. Oversteered / Rambling Models Marked `INCONCLUSIVE`** | A high $\alpha$ that makes the model loop into the 1,024-token ceiling raises variance and gets misclassified as `INCONCLUSIVE` ("needs more samples"). | **Explicit `BROKEN` Verdict:** Any arm whose token truncation rate or answer-parse failure rate exceeds `base` by $> +1.0\%$ is immediately flagged as **`BROKEN (Oversteered)`**. |
| **5. Teacher-Rewriting Confounder** | Relying on GPT-5.1 to create dense traces requires paid APIs and conflates teacher distillation with activation steering. | **Dual Vector Arms:** Compares both **On-Policy Rule-Compressed Traces** (`steered_onpolicy`, zero external API) and **Teacher-Compressed Traces** (`steered_teacher`). |
| **6. Cold Outreach Ghosting Risk** | Sending 10 cold DMs and expecting 5 replies ($50\%$ reply rate) in 72 hours fails due to normal inbox ghosting. | **25 Targeted DMs + 2 Community Threads:** Expands top-of-funnel to 25 active steering researchers/contributors plus open threads in `r/LocalLLaMA` and EleutherAI `#interpretability`. |
| **7. Unsupported Ecosystem Claims** | Claiming `boost ollama` when Ollama `Modelfile` only supports `FROM`, `PARAMETER`, `TEMPLATE`, `SYSTEM`, `ADAPTER` (LoRA), `LICENSE`, `MESSAGE`, and `REQUIRES`. | **Hard Scope Lock:** Zero Ollama or LM Studio promises. Pure PyTorch hook evaluator first; `llama.cpp --control-vector` export is built only post-Gate 2 if `GO` is achieved and an external user requests it. |

---

## 3. STEP 0 & WEEK 0: PRACTITIONER VALIDATION SPRINT

### 3.1 Step 0: Locked Project Goal
* **Chosen Path — Goal A + B Combined (Replication Note + Reproducible Benchmark Repo):**
  * **Goal A (Scientific / Community Impact):** Publish a transparent, fully reproducible replication report testing activation-steering claims on `Qwen2.5-1.5B-Instruct` and `Llama-3.2-1B-Instruct`, including positive, negative, and control arms.
  * **Goal B (Open-Source Engineering Artifact):** Release the minimal verification harness (`steergate`) and complete `.jsonl` rollout logs so any researcher can verify the confidence intervals in one command.
* **Explicitly Cut — Goal C (Monetization / SaaS):** Zero paid cartridge tiers, zero web dashboards, zero cloud hosting.

### 3.2 Week 0: Demand Validation Sprint (Days 1–5, Parallel with Gate 1)
1. **Outreach Target List (25 Direct Messages + 2 Forum Posts):**
   * **10 Authors/Maintainers** of recent steering repos and papers (`repeng`, `DenseSteer`, `AxBench`, `steering-vectors`, `CAA`, `pyvene`).
   * **10 Contributors/Users** who opened issues, PRs, or discussions on `llama.cpp` `--control-vector` and `llama-control-vector-generator`.
   * **5 Local-LLM Practitioners** publishing steered/abliterated models on HuggingFace and `r/LocalLLaMA`.
   * **2 Public Posts** on `r/LocalLLaMA` and the EleutherAI Discord `#interpretability` channel.
2. **The Outreach Message:**
   > **Subject:** quick question on separating steering-vector gains from sampling noise
   >
   > Hi `<Name>`,
   >
   > I'm replicating small activation-steering gains on GSM8K ($+1$ to $+2$ points on `Qwen2.5-1.5B-Instruct`) and finding that sampling variance, layer cherry-picking, and simple prompt baselines make it hard to tell if a vector's lift is real.
   >
   > I'm building a tiny open-source checker (`steergate`): paired per-question seeded rollouts ($k=8$), random-direction and prompt-only controls at equal norm, and a single `GO / NO-GO / INCONCLUSIVE / BROKEN` verdict with a paired bootstrap confidence interval.
   >
   > How do you currently verify that a steering vector helps without hurting fluency, and would you run a lightweight checker like that on your own vectors? Happy to share our raw replication logs once Gate 2 finishes this week.
3. **Gate 3 Pass / Pivot Rule:**
   * **PASS:** $\ge 5$ total replies, with $\ge 3$ describing the evaluation/noise problem unprompted AND $\ge 2$ asking to try the tool $\rightarrow$ build `steergate report` and `steergate gate` in Days 11–21.
   * **FAIL:** $< 2$ want to run a standalone CLI tool $\rightarrow$ drop all CLI packaging and spend Days 11–28 purely on **Goal A** (expanding the replication paper across 3 steering recipes and 2 model families).

---

## 4. THE FIVE KILL GATES (FAIL CHEAPLY BEFORE BUILDING)

Each gate has a strict time box, quantitative pass bar, and automatic kill/pivot trigger. No downstream code is written until the current gate passes.

| Gate | Window | Exact Test Executed | Quantitative Pass Bar | If It Fails (Kill / Fix Rule) |
| :--- | :--- | :--- | :--- | :--- |
| **G1: Dry Run & Greedy Baseline (`P0-a`)** | Days 1–2 | 1. 20-question smoke test (`T=0`) on `Qwen/Qwen2.5-1.5B-Instruct`.<br>2. Full 1,319-question GSM8K test set greedy baseline (`T=0`, `max_new_tokens=1024`). | • Greedy accuracy within **$60.0\%\text{--}66.0\%$**.<br>• Regex answer parser succeeds on $\ge 98.5\%$ of outputs.<br>• 50-question re-run yields $100\%$ identical token IDs. | Fix chat template (`apply_chat_template`) or regex parser (max **2 attempts**). If still outside $60.0\%\text{--}66.0\%$, **STOP** and fix environment. |
| **G1.5: Out-of-Sample Hyperparameter Lock** | Day 3 | 1. Extract vectors on `GSM8K-train[0:150]` across middle layers $L \in \{12, 14, 16, 18, 20\}$ (of 28).<br>2. Sweep $(L, \alpha)$ with $\alpha \in \{0.5, 1.0, 1.5, 2.0\}$ on `GSM8K-train[150:350]` ($N=200$) + compute token KL divergence on 50 WikiText-2 passages. | • Lock single $(L^*, \alpha^*)$ maximizing train accuracy subject to $D_{\text{KL}} \le 0.15\text{ nats/token}$ and truncation $\le \text{Base} + 0.5\%$.<br>• Save SHA-256 hash of vectors and $(L^*, \alpha^*)$ to `locked_config.json`. | If all $(L, \alpha)$ degrade train accuracy, lock the least-harmful configuration at $60\%$ depth ($L=17, \alpha=1.0$) to formally test and document a `NO-GO` on the test set. |
| **G2-a: Empirical Noise Floor (`P0-b` & `P0-c`)** | Days 4–5 | 1. **`P0-b` (Seed Drift):** Run `base` arm twice on all 1,319 test questions ($k=8, T=0.7$) with Master Seed A vs. Master Seed B.<br>2. **`P0-c` (Hardware/Dtype Drift):** Compare 300-question stratified slice across two GPUs/dtypes (`T4 fp16` vs. `P100/L4 bf16`). | • Paired 95% CI between Seed A and Seed B covers $0.0\%$.<br>• Empirical noise floor $d_0 = |\hat{\Delta}_{\text{null}}|$ is $\le 0.5\text{ percentage points}$. | Bug in CRN seeding or answer parser: fix once and re-run before starting steered arms. |
| **G2-b: Full 6-Arm Locked Replication (`P0-d`)** | Days 6–10 | Run all 6 locked arms on the full 1,319 GSM8K test set ($k=8, T=0.7, \text{top-}p=0.8$) using `locked_config.json`. | Outputs an unambiguous `GO`, `NO-GO`, `INCONCLUSIVE`, or `BROKEN` verdict with 10,000-sample paired bootstrap CIs. | If Kaggle session times out, resume from `.jsonl` checkpoint. Never modify `locked_config.json`. |
| **G3: Demand Validation** | Days 1–7 | Tabulate replies from the 25 cold DMs + 2 community posts. | $\ge 3$ confirm pain unprompted AND $\ge 2$ request the CLI checker. | Pivot to **Goal A Only** (publish replication study + raw scripts; skip CLI/Action). |

---

## 5. FROZEN MATHEMATICAL & STATISTICAL PROTOCOL

### 5.1 Three-Way Disjoint Data Split (Zero Test Leakage)
1. **Extraction Split ($D_{\text{extract}}$): `GSM8K-train[0:150]` ($150$ problems)**
   * Generate on-policy verbose rollouts $y_i^-$ from `Qwen2.5-1.5B-Instruct` (retaining problems where the model reaches the correct final answer).
   * Construct two positive trace sets to isolate teacher distillation from representation steering:
     * **`onpolicy` ($y_i^{+,\text{self}}$):** Deterministic algebraic compression of the model's own trace $y_i^-$ (stripping conversational filler phrases while keeping the model's own equations intact—requiring zero external LLM API calls).
     * **`teacher` ($y_i^{+,\text{teach}}$):** Dense traces rewritten following the DenseSteer protocol.
2. **Tuning & Fluency Split ($D_{\text{tune}}$): `GSM8K-train[150:350]` ($200$ problems) + `WikiText-2[0:50]`**
   * Used exclusively in **Gate 1.5** to lock $(L^*, \alpha^*)$ under the fluency ceiling $D_{\text{KL}}(P_{\text{base}} \parallel P_{L,\alpha}) \le 0.15\text{ nats/token}$.
3. **Evaluation Split ($D_{\text{test}}$): Full `GSM8K-test[0:1319]` ($1,319$ problems)**
   * Untouched until **Gate 2**.

---

### 5.2 Contrastive Vector Extraction & Equal-Norm Scaling
For each candidate middle layer $L \in \{12, 14, 16, 18, 20\}$ and pair $(x_i, y_i^+, y_i^-) \in D_{\text{extract}}$, capture the residual-stream activation $\mathbf{h}_L \in \mathbb{R}^{d_{\text{model}}}$ at the final token position $T_{\text{last}}$:

$$\vec{\mu}_L^+ = \frac{1}{|D_{\text{extract}}|} \sum_{i=1}^{|D_{\text{extract}}|} \mathbf{h}_L(x_i \oplus y_i^+)_{T_{\text{last}}}, \qquad \vec{\mu}_L^- = \frac{1}{|D_{\text{extract}}|} \sum_{i=1}{|D_{\text{extract}}|} \mathbf{h}_L(x_i \oplus y_i^-)_{T_{\text{last}}}$$

Normalize the difference vector to unit $L_2$ length and multiply by the average residual-stream token norm $\bar{r}_L$ on $D_{\text{extract}}$ so that scale $\alpha$ represents a dimensionless percentage of layer $L$'s native activation magnitude:

$$\hat{v}_L = \frac{\vec{\mu}_L^+ - \vec{\mu}_L^-}{\|\vec{\mu}_L^+ - \vec{\mu}_L^-\|_2 + 10^{-8}}, \qquad \vec{v}_L^{\text{scaled}} = \left(\frac{\alpha^*}{100} \cdot \bar{r}_L\right) \hat{v}_L$$

During inference at layer $L^*$, the forward hook adds $\vec{v}_{L^*}^{\text{scaled}}$ in-place:

$$\mathbf{h}_{L^*}^{\text{steered}} = \mathbf{h}_{L^*} + \vec{v}_{L^*}^{\text{scaled}}$$

---

### 5.3 Solving the PyTorch Batch-Seeding Bug: Step-Indexed Gumbel-Max CRN
HuggingFace's `model.generate()` does **not** support passing an array of independent `torch.Generator` objects per batch row, and a shared generator drifts across batch items whenever one question generates an extra token.

SteerGate solves this cleanly using the **Step-Indexed Gumbel-Max Trick** inside a 25-line custom batched token loop:
1. For Question $q \in \{0, \dots, 1318\}$, Rollout $k \in \{0, \dots, 7\}$, and Generation Step $t \in \{0, \dots, 1023\}$, compute a deterministic 32-bit seed via `zlib.crc32(f"{master_seed}:{q}:{k}:{t}".encode())`.
2. After applying temperature $T = 0.7$ and $\text{top-}p = 0.8$ filtering to the vocabulary log-probabilities $\log \mathbf{p}_{q,k,t}$, draw uniform random noise $\mathbf{u}_{q,k,t} \sim \text{Uniform}(0, 1)^{|\mathcal{V}_{\text{top-}p}|}$ seeded strictly by `(master_seed, q, k, t)` and select the next token via the Gumbel-Max identity:

$$\text{next\_token}_{q,k,t} = \arg\max_{v \in \mathcal{V}_{\text{top-}p}} \Big( \log p_{q,k,t}(v) - \log\big(-\log u_{q,k,t}(v)\big) \Big)$$

**Why This Is Mathematically Exact:**
* It is $100\%$ mathematically equivalent to multinomial sampling from the temperature-scaled, $\text{top-}p$-filtered distribution.
* If `base` and `steered` produce identical logits for the first 40 tokens of Question #412 Rollout #3, they are guaranteed to generate the **exact same token sequence** up to step 40, regardless of whether `batch_size` is $1$, $16$, or $64$.

---

### 5.4 The 6 Mandatory Evaluation Arms

| Arm Name | Prompt Used | Layer $L^*$ Vector Injected | Scientific Purpose |
| :--- | :--- | :--- | :--- |
| **`1. base`** | Standard CoT prompt | None ($\alpha = 0$) | Baseline accuracy of untouched `Qwen2.5-1.5B-Instruct`. |
| **`2. prompt_dense`** | Standard CoT + 1-line dense-equation instruction | None ($\alpha = 0$) | **AxBench Prompt Baseline:** Tests if a 1-line prompt matches or beats steering for free. |
| **`3. steered_onpolicy`** | Standard CoT prompt | $+\vec{v}_{L^*}^{\text{self}}$ (from self-compressed traces) | **Zero-Teacher Steering:** Tests if steering works without GPT-5.1 teacher distillation. |
| **`4. steered_teacher`** | Standard CoT prompt | $+\vec{v}_{L^*}^{\text{teach}}$ (from DenseSteer-style traces) | **Published Replication Arm:** Directly tests the DenseSteer recipe under paired controls. |
| **`5. ctrl_random`** | Standard CoT prompt | $+\vec{u}_{L^*}^{\text{rand}}$ with $\|\vec{u}_{L^*}^{\text{rand}}\|_2 = \|\vec{v}_{L^*}\|_2$ | **Random Direction Control:** Rules out stochastic resonance / logit noise effects. |
| **`6. ctrl_shuffled`** | Standard CoT prompt | $+\vec{v}_{L^*}^{\text{shuf}}$ (permuted $+$ / $-$ labels, equal $L_2$ norm) | **Shuffled-Pair Control:** Rules out dataset token-frequency artifacts. |

---

### 5.5 Question-Clustered Paired Bootstrap & Four-Way Verdict Rule
For each question $i \in \{1, \dots, 1319\}$ and arm $A$, compute the mean pass rate across the $k = 8$ rollouts:
$$s_{i,A} = \frac{1}{8} \sum_{j=1}^{8} \mathbb{I}\big(\text{pred}_{i,j,A} = \text{gold}_i\big)$$
For any candidate steered arm $S \in \{\text{steered\_onpolicy}, \text{steered\_teacher}\}$ and reference arm $R \in \{\text{base}, \text{prompt\_dense}, \text{ctrl\_random}, \text{ctrl\_shuffled}\}$, compute the per-question paired difference $d_i^{(S,R)} = s_{i,S} - s_{i,R}$ and draw $B = 10,000$ question-clustered bootstrap resamples to obtain the point lift $\hat{\Delta}^{(S,R)}$, two-sided 95% CI $[\Delta_{\text{low}, 2.5\%}^{(S,R)}, \Delta_{\text{high}, 97.5\%}^{(S,R)}]$, and one-sided 95% upper bound $\Delta_{\text{high}, 95\%}^{(S,R)}$:

1. **`BROKEN (Oversteered)`:**
   $$\text{TruncRate}_S > \text{TruncRate}_{\text{base}} + 1.0\% \quad \text{OR} \quad \text{ParseFail}_S > \text{ParseFail}_{\text{base}} + 1.0\%$$
2. **`GO (Verified Gain)`:**
   Not `BROKEN`, AND point lift $\hat{\Delta}^{(S,\text{base})} \ge +1.0\%$, AND lower 95% CI bounds satisfy:
   $$\Delta_{\text{low}, 2.5\%}^{(S,\text{base})} > 0, \quad \Delta_{\text{low}, 2.5\%}^{(S,\text{ctrl\_random})} > 0, \quad \text{and} \quad \Delta_{\text{low}, 2.5\%}^{(S,\text{ctrl\_shuffled})} > 0$$
   *(Sub-classified as **`GO-SUPERIOR`** if $\Delta_{\text{low}, 2.5\%}^{(S,\text{prompt\_dense})} > 0$, or **`GO-REDUNDANT`** if `prompt_dense` matches or beats $S$).*
3. **`NO-GO (Null / Debunked)`:**
   Not `BROKEN`, AND either the one-sided 95% upper bound rules out a $+1.0\%$ gain ($\Delta_{\text{high}, 95\%}^{(S,\text{base})} < +1.0\%$) OR the vector fails to beat random/shuffled controls ($\hat{\Delta}^{(S,\text{ctrl\_random})} \le 0$ or $\hat{\Delta}^{(S,\text{ctrl\_shuffled})} \le 0$).
4. **`INCONCLUSIVE`:**
   Not `BROKEN`, $\Delta_{\text{high}, 95\%}^{(S,\text{base})} \ge +1.0\%$, but $\Delta_{\text{low}, 2.5\%}^{(S,\text{base})} \le 0$.

---

## 6. KAGGLE T4x2 COMPUTE BUDGET & EXECUTABLE DAY-1 CODE

### 6.1 Exact Compute & Wall-Clock Arithmetic on Free Kaggle GPUs
Kaggle provides **30 hours/week of free dual NVIDIA T4 GPUs (`T4 x2`, $2 \times 16\text{ GB}$ VRAM)**.
* **Model Footprint:** `Qwen2.5-1.5B-Instruct` in `float16` takes $3.1\text{ GB}$ VRAM per GPU, leaving $12.9\text{ GB}$ for KV-cache.
* **Throughput per T4 GPU:** With `batch_size = 32` (4 questions $\times$ $k = 8$ rollouts per batch) and average GSM8K completion length of $240\text{ tokens}$, one T4 generates $\approx 1,100\text{ tokens/sec}$, completing 1 batch ($32\text{ rollouts}$) every $\approx 7.0\text{ seconds}$.
* **Time per Arm on Dual T4 (`T4 x2` Data Parallel):**
  $$\frac{1,319\text{ questions} \times 8\text{ rollouts}}{32\text{ batch size} \times 2\text{ GPUs}} = 165\text{ batches per GPU} \times 7.0\text{ sec} \approx 19.3\text{ minutes per arm}$$
* **Total Gate 2 Compute Budget:** Running all 6 arms + the `G2-a` noise-floor check ($7\text{ full passes}$ total) takes **$7 \times 19.3\text{ min} \approx 2.25\text{ hours}$ on Kaggle `T4 x2`**—less than $8\%$ of a single week's free Kaggle quota!

---

### 6.2 Phase 1 Repository Layout (Days 1–10: Under 300 Lines of Code)

```text
steergate/
├── data/
│   ├── gsm8k_train_150_pairs.jsonl   # 150 (question, y_sparse, y_self_dense, y_teacher_dense)
│   └── locked_config.json            # Written by Gate 1.5 (L*, alpha*, SHA-256 hashes)
├── vectors/
│   └── qwen2.5_1.5b_vectors.safetensors
├── logs/
│   ├── g1_greedy.jsonl               # Gate 1 output (1,319 greedy rollouts)
│   └── g2_paired_runs.jsonl          # Gate 2 output (resumable append-only log)
├── g1_baseline.py                    # Day 1-2: 20-q dry run + 1,319 greedy baseline
├── g1_5_lock.py                      # Day 3: Extract vectors + lock (L*, alpha*) on train[150:350]
└── g2_runner.py                      # Day 4-10: Dual-GPU CRN runner + 10,000-bootstrap verdict
```

---

### 6.3 Ready-to-Run Gate 1 & CRN Core (`g1_baseline.py` Reference Implementation)
This self-contained core implements exact GSM8K answer extraction, append-only `.jsonl` crash recovery, and the step-indexed CRN Gumbel-Max sampler:

```python
import hashlib, json, os, re, zlib
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = "Solve the following math problem step by step. Put your final numerical answer at the very end after ####."

def extract_gsm8k_answer(text: str) -> tuple[str | None, bool]:
    """Extracts answer after #### or \\boxed{}, falling back to the final number."""
    if "####" in text:
        after = text.split("####")[-1].strip()
        m = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", after)
        if m:
            return m.group(0).replace(",", ""), False
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        m = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", boxed[-1])
        if m:
            return m.group(0).replace(",", ""), False
    nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if nums:
        return nums[-1].replace(",", ""), False
    return None, True

def step_crn_sample(logits: torch.Tensor, qids: list[int], ks: list[int], step: int,
                    master_seed: int, temp: float = 0.7, top_p: float = 0.8) -> torch.Tensor:
    """Batch-invariant Common Random Numbers (CRN) sampler via Step-Indexed Gumbel-Max."""
    if temp == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    scaled = logits / temp
    sorted_logits, sorted_indices = torch.sort(scaled, descending=True, dim=-1)
    cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    remove_mask = (cum_probs - torch.softmax(sorted_logits, dim=-1)) >= top_p
    sorted_logits[remove_mask] = float("-inf")
    filtered_log_probs = torch.log_softmax(
        torch.zeros_like(scaled).scatter(-1, sorted_indices, sorted_logits), dim=-1
    )
    next_tokens = []
    for row_idx, (q, k) in enumerate(zip(qids, ks)):
        seed = zlib.crc32(f"{master_seed}:{q}:{k}:{step}".encode("utf-8")) & 0xFFFFFFFF
        gen = torch.Generator(device=logits.device).manual_seed(seed)
        u = torch.empty_like(filtered_log_probs[row_idx]).uniform_(1e-6, 1.0 - 1e-6, generator=gen)
        gumbel = -torch.log(-torch.log(u))
        tok = torch.argmax(filtered_log_probs[row_idx] + gumbel, dim=-1)
        next_tokens.append(tok)
    return torch.stack(next_tokens).unsqueeze(-1)
```

---

## 7. 30-DAY SOLO EXECUTION ROADMAP & HARD GUARDRAILS

| Days | Milestone / Gate | Concrete Engineering & Experimental Actions | Exit Artifact |
| :--- | :--- | :--- | :--- |
| **Days 1–2** | **Gate 1 (`P0-a`) + Launch Week 0 Outreach** | • Send 25 cold DMs and post 2 forum threads.<br>• Run 20-question dry run on `Qwen2.5-1.5B-Instruct`, then full 1,319 greedy baseline. Verify accuracy is within $60.0\%\text{--}66.0\%$. | `logs/g1_greedy.jsonl` |
| **Day 3** | **Gate 1.5 (Out-of-Sample Hyperparameter Lock)** | • Build 150-pair contrastive set on `GSM8K-train[0:150]`.<br>• Sweep $L \in \{12, 14, 16, 18, 20\}$ and $\alpha \in \{0.5, 1.0, 1.5, 2.0\}$ on `GSM8K-train[150:350]` + 50 WikiText passages ($D_{\text{KL}} \le 0.15$). | `data/locked_config.json` (SHA-256 locked) |
| **Days 4–5** | **Gate 2-a (`P0-b/c` Noise Floor)** | • Run `base` arm twice with Master Seeds `101` vs. `202` ($k=8$) on Kaggle `T4 x2` (~38 mins total) to lock empirical noise floor $d_0$.<br>• Verify cross-GPU consistency (`T4 fp16` vs. `P100/L4 bf16`). | `logs/g2a_noise_floor.json` |
| **Days 6–10** | **Gate 2-b (`P0-d` Full 6-Arm Sweep) & Gate 3 Fork** | • Run all 6 locked arms across all 1,319 GSM8K test problems ($k=8$, ~2 hours on `T4 x2`).<br>• Compute 10,000-sample paired bootstrap CIs and output the 4-way verdict.<br>• Score Week 0 outreach replies against the Gate 3 pass bar. | `logs/g2b_verdict.json` + Gate 3 Decision |
| **Days 11–21** | **Post-Gate Fork (Path 1 vs. Path 2)** | • **If Gate 3 Passed:** Build `steergate report` and `steergate gate` CLI commands around the validated runner.<br>• **If Gate 3 Failed:** Skip CLI packaging; evaluate PCA-extracted vectors and TruthfulQA sycophancy vectors using the raw harness. | `steergate` CLI **OR** Expanded 3-Recipe Dataset |
| **Days 22–28** | **Second Model Family & Replication Paper** | • Run the identical G1 $\rightarrow$ G1.5 $\rightarrow$ G2 pipeline on `meta-llama/Llama-3.2-1B-Instruct` (or `Qwen2.5-3B-Instruct`).<br>• Write the complete replication study (`REPLICATION.md`) logging every positive, negative, and control result. | Second model logs + `REPLICATION.md` |
| **Days 29–30** | **External Reproduction & Public Release** | • Have at least 1 outside practitioner reproduce the verdict table from `logs/g2_paired_runs.jsonl`.<br>• Publish repo, raw `.jsonl` logs, and write-up on GitHub, `r/LocalLLaMA`, and X. | Verified v1.0 Public Release |

### Non-Negotiable Scope & Integrity Rules
1. **No Unverified Claims:** Never claim a vector improves reasoning until `g2b_verdict.json` outputs `GO`.
2. **Frozen Config Mandate:** Once `locked_config.json` is hashed on Day 3, altering $L^*$, $\alpha^*$, prompts, or regex logic on the 1,319 test set is strictly forbidden. Any follow-up tuning must be logged in a separate post-hoc appendix.
3. **No Feature Creep:** No web UI, no Ollama `Modelfile` claims, no paid tiers. Only add `llama.cpp --control-vector` export after Day 30 if Gate 2 = `GO` and an outside user asks for it.
4. **The Only 3 Success Metrics That Count:**
   * **Metric 1:** At least **1 outside person** reproduces a gate verdict from your published `.jsonl` file.
   * **Metric 2:** At least **3 steering claims** (`steered_onpolicy`, `steered_teacher`, and a second trait/estimator) are evaluated against `prompt_dense` and `ctrl_random` and logged publicly.
   * **Metric 3:** Verdicts agree across **2 hardware/dtype environments** within the empirical noise floor $d_0$.