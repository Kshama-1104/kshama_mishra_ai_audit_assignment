# Lab Notebook

## Timebox Summary

This assignment was handled as a focused hiring-assignment audit over several calendar days, with about 10 focused hours of work prioritized around evidence quality rather than volume.

- Day 1, July 23: read the assignment, understood the required deliverables, and planned the repository structure.
- Day 2, July 24: set up Python, VS Code, tokenizer/ML packages, Git, and the local project structure.
- Day 3, July 25: inspected the starter kit, reproduced starter outputs, audited Part B, drafted Part C, and began Part A tokenizer investigation.
- Day 4, July 26: built the FLORES-200 corpus, ran corrected tokenizer analysis, strengthened measured evidence, added reproducibility scripts, and polished reports.
- Day 5, July 27: final report polish, reproducibility check, repository review, and GitHub push preparation.

Some commits were made after several related edits were completed, so the Git history is organized by meaningful milestones rather than every small notebook action.

## 2026-07-23 - Day 1 - Assignment Reading and Initial Plan

### Goal

Audit the provided tokenizer and serving-stack report with measured evidence.

### Initial understanding

The assignment asks for an audit, not just a rewrite of the previous report. The important rule is that every claimed flaw needs evidence: isolate the issue, measure its effect, and state the direction and magnitude of the distortion.

### Planned deliverables

- `NOTEBOOK.md` for work log and dead ends.
- `AI_USAGE.md` for honest AI assistance disclosure.
- `partA/tokenizer_audit.md` for tokenizer audit.
- `partB/capacity_reconciliation.md` for serving/capacity reconciliation.
- `partC/memo.md` for decision memo.
- `README.md` for repository overview and reproduction steps.

### Initial plan

I planned to inspect the starter files first, then begin with Part B because it is self-contained and depends only on the model spec and benchmark log. After that, I would complete Part C and then return to Part A, which requires corpus construction and tokenizer experiments.

### Starter files to inspect

- `starter_kit/REPORT_v0.md`
- `starter_kit/fertility.py`
- `starter_kit/bench/model_spec.md`
- `starter_kit/bench/bench_log.csv`

## 2026-07-24 - Day 2 - Environment and Repository Setup

### Setup completed

- Created a clean assignment repository structure.
- Installed Python 3.11.9.
- Installed required analysis/tokenizer packages.
- Verified `pip`, Jupyter, tokenizer libraries, and ML packages.
- Checked local machine capacity and confirmed that local CPU execution was enough for this audit.
- Initialized Git and prepared the first local commit.

### Environment notes

The laptop has limited RAM and no CUDA GPU, so I avoided unnecessary local model training. The assignment is mainly analysis, scripting, and evidence-based reporting, so VS Code plus local Python was sufficient. Colab remained a fallback option if GPU-heavy work became necessary.

### Repository organization

I kept the original starter kit unchanged for auditability and created separate folders for my answers:

- `partA/`
- `partB/`
- `partC/`
- `starter_kit/`

## 2026-07-25 - Day 3 - Starter File Inspection

### Files inspected

- `starter_kit/REPORT_v0.md`
- `starter_kit/fertility.py`
- `starter_kit/bench/model_spec.md`
- `starter_kit/bench/bench_log.csv`

### Initial observations

- `REPORT_v0.md` contains the previous intern's conclusions that need to be audited.
- `fertility.py` is the script behind the tokenizer numbers and needs code/metric review.
- `model_spec.md` contains model details needed for KV-cache calculations.
- `bench_log.csv` contains the serving benchmark rows needed for Part B.

### Next step

Start Part B1 by loading the benchmark log and extracting the model parameters from `model_spec.md`.

## 2026-07-25 - Day 3 - Part B1 KV-cache Calculation

### Hypothesis

The maximum number of concurrent 4096-token sequences should be limited mainly by KV-cache memory.

### Experiment / calculation

From `starter_kit/bench/model_spec.md`:

- layers = 28
- KV heads = 8
- head_dim = 128
- KV cache precision = fp16 = 2 bytes
- GPU memory = 24 GB
- gpu_memory_utilization = 0.92
- non-KV runtime overhead = 1.6 GB
- max_model_len = 4096

KV-cache bytes per token:

`28 * 2 * 8 * 128 * 2 = 114,688 bytes/token`

The factor of 2 is for K and V.

Usable GPU memory:

`24 GB * 0.92 = 22.08 GB`

After runtime overhead:

`22.08 GB - 1.6 GB = 20.48 GB`

KV memory per 4096-token sequence:

`114,688 * 4096 = 469,762,048 bytes`

Approximate max concurrent 4096-token sequences:

`20.48 GB / 469,762,048 bytes ≈ 46 sequences`

### Result

The model should fit roughly 46 full 4096-token sequences before KV-cache memory becomes the bottleneck.

### Revision / next step

Check this prediction against `bench_log.csv`, especially the rows where `prompt_len + gen_len = 4096` and batch size approaches or exceeds this range.

## 2026-07-25 - Day 3 - Part B2-B4 Throughput and Goodput Audit

### Hypothesis

The report's throughput conclusion may be wrong because `reported_tok_s` may include prompt/prefill tokens instead of only generated output tokens.

### Experiment

I loaded `starter_kit/bench/bench_log.csv` and computed output-token goodput as:

`num_requests * gen_len / wall_clock_s`

For the long-context rows, `prompt_len = 3584` and `gen_len = 512`, so each request reaches 4096 total tokens.

### Result

For the long-context sweep:

- batch 16: goodput = 163.94 output tok/s, preempted_seqs = 0, kv_cache_util = 0.62
- batch 24: goodput = 200.92 output tok/s, preempted_seqs = 0, kv_cache_util = 0.93
- batch 32: goodput = 172.99 output tok/s, preempted_seqs = 7, kv_cache_util = 0.97
- batch 48: goodput = 162.31 output tok/s, preempted_seqs = 23, kv_cache_util = 0.97

The anomaly is that useful output-token goodput peaks at batch 24 and then drops, even though batch size increases.

### Revision / conclusion

The likely mechanism is KV-cache saturation causing scheduler preemptions. I will recommend capping long-context batch size around 24 unless serving config changes increase available KV-cache capacity.

## 2026-07-25 - Day 3 - Part C Decision Memo

### Hypothesis

Prompt engineering may be the best first launch path because the task is style control, the deadline is short, and reviewer coverage is limited.

### Reasoning

I compared three paths: prompt engineering, synthetic SFT, and a small inference-time rewriter. Prompt engineering is fastest and cheapest to test. SFT is feasible on one A100-80GB but risky because synthetic data quality and multilingual review coverage are the bottlenecks. A rewriter adds latency and can alter factual content.

### Result

I chose prompt engineering plus a small evaluation set as the first path, with a kill criterion by day 3. If prompt-only fails to reach a useful Hindi/Kannada naturalness threshold without correctness regressions, I would switch to synthetic SFT.

### Revision / next step

Commit Part C, then begin Part A tokenizer audit.

## 2026-07-25 - Day 3 - Part A Initial fertility.py Smoke Test

### Hypothesis

Before changing anything, I need to reproduce the previous script's behavior on the provided tiny sample corpus.

### Experiment

I first ran the script without arguments:

`python starter_kit/fertility.py`

It failed with the expected argparse error because `--corpus` is required.

I then accidentally passed two corpora after one `--corpus` flag:

`python starter_kit/fertility.py --corpus eng=starter_kit/corpus_sample/eng_sample.txt hin=starter_kit/corpus_sample/hin_sample.txt`

This failed because the CLI expects `--corpus` to be repeated once per corpus. This was a usage mistake, not a script bug.

Correct command:

`python starter_kit/fertility.py --corpus eng=starter_kit/corpus_sample/eng_sample.txt --corpus hin=starter_kit/corpus_sample/hin_sample.txt`

I also verified that explicitly setting the default tokenizer gives the same result:

`python starter_kit/fertility.py --corpus eng=starter_kit/corpus_sample/eng_sample.txt --corpus hin=starter_kit/corpus_sample/hin_sample.txt --tokenizer gpt2`

### Result

Both correct commands produced:

- English fertility: 1.27 tok/word
- Hindi fertility: 7.45 tok/word
- Hindi/English fertility ratio: 5.89x
- English tok/char: 0.226
- Hindi tok/char: 1.579

### Revision / next step

The first suspicious behavior was not a bug: `--corpus` is intentionally repeatable. Next I will audit the code for real metric and implementation issues, then measure their effect.

## 2026-07-25 - Day 3 - Part A Metric Aggregation Check

### Hypothesis

The script may distort tokenizer fertility because it averages per-line ratios instead of computing total tokens divided by total words/chars. If line lengths differ, short lines get the same weight as long lines.

### Experiment

I compared the current script-style line average against an aggregate calculation on the starter sample corpus.

Current script-style fertility:

`mean(tokens_per_line / words_per_line)`

Aggregate fertility:

`sum(tokens) / sum(words)`

### Result

On the starter sample:

| language | current line avg tok/word | aggregate tok/word | current line avg tok/char | aggregate tok/char |
| -------- | ------------------------: | -----------------: | ------------------------: | -----------------: |
| eng      |                    1.2652 |             1.2692 |                    0.2256 |             0.2210 |
| hin      |                    7.4485 |             7.5246 |                    1.5791 |             1.5828 |

The Hindi/English tok/word ratio changes from about 5.887x to 5.929x.

### Revision / conclusion

This is a real metric implementation problem, but its effect on the tiny starter sample is small. It could matter more on a real corpus with more variable line lengths.

## 2026-07-25 - Day 3 - Part A Whitespace Splitting Check

### Hypothesis

The script may miscount words because it uses `line.split(" ")` instead of `line.split()`. Splitting only on a literal space can create empty words for double spaces and fails to split tabs.

### Experiment

I tested example strings with normal spaces, double spaces, tabs, and Hindi text:

`line.strip().split(" ")` versus `line.strip().split()`

### Result

Examples:

| input                          | `split(" ")` count | `split()` count | issue                           |
| ------------------------------ | -------------------: | ----------------: | ------------------------------- |
| `hello world`                |                    2 |                 2 | none                            |
| `hello  world`               |                    3 |                 2 | double space creates empty word |
| `hello\tworld`               |                    1 |                 2 | tab is not treated as separator |
| `नमस्ते  दुनिया` |                    3 |                 2 | double space creates empty word |

### Revision / conclusion

This is a real code bug. The script should use `line.split()` for whitespace tokenization. The current implementation can inflate or deflate the word denominator depending on whitespace formatting, distorting `tok/word`.

## 2026-07-25 - Day 3 - Part A Denominator Concept Check

### Hypothesis

Even if the code computed `tokens / whitespace word` correctly, that denominator may be conceptually wrong for cross-language routing and cost decisions.

### Reasoning

Whitespace words are not comparable across languages. Languages differ in morphology, script, spacing conventions, and what counts as a "word". For routing/cost, the question is not "tokens per whitespace word"; it is "how many tokens will this user request consume for the same underlying content?"

### Result

A better decision metric should hold content constant. For a parallel corpus, `tokens per parallel sentence` is closer to the routing/cost question because each row expresses the same semantic content across languages. Secondary denominators such as UTF-8 bytes or grapheme clusters can help diagnose tokenizer behavior, but `tok/word` should not be the main routing metric.

### Revision / conclusion

I will treat `tok/word` as a diagnostic metric only. For the corrected analysis, I will report tokens per parallel sentence and at least one lower-level denominator such as tokens per UTF-8 byte or grapheme cluster.

## 2026-07-26 - Day 4 - Part A FLORES Corpus Construction

### Hypothesis

The starter corpus is too small for reliable tokenizer conclusions, so I need a larger multilingual parallel corpus with English, Hindi, and Dravidian languages.

### Experiment

I first tried `facebook/flores` through Hugging Face, but it was gated and required authentication. I then tried the open `Muennighoff/flores200` mirror. The installed `datasets` version no longer supported dataset scripts, so I downgraded to `datasets==2.21.0`. On Windows, the dataset loader also needed UTF-8 mode, so I used `python -X utf8`.

I created corpus files from the FLORES-200 dev split for:

- English: `eng_Latn`
- Hindi: `hin_Deva`
- Kannada: `kan_Knda`
- Tamil: `tam_Taml`
- Telugu: `tel_Telu`

### Result

Created:

| file                               | language | FLORES code  | lines |
| ---------------------------------- | -------- | ------------ | ----: |
| `partA/corpus_flores200/eng.txt` | English  | `eng_Latn` |   997 |
| `partA/corpus_flores200/hin.txt` | Hindi    | `hin_Deva` |   997 |
| `partA/corpus_flores200/kan.txt` | Kannada  | `kan_Knda` |   997 |
| `partA/corpus_flores200/tam.txt` | Tamil    | `tam_Taml` |   997 |
| `partA/corpus_flores200/tel.txt` | Telugu   | `tel_Telu` |   997 |

### Revision / conclusion

The corpus now satisfies the assignment requirement: at least four languages including English, Hindi, and two Dravidian languages. I included three Dravidian languages: Kannada, Tamil, and Telugu.

## 2026-07-26 - Day 4 - Part A Corrected Tokenizer Analysis

### Hypothesis

A multilingual tokenizer should reduce the extreme token inflation seen with GPT-2 on Indic languages. The best routing/cost metric should be tokens per parallel sentence, because each FLORES row represents aligned semantic content.

### Experiment

I compared two tokenizers on the 997-line FLORES corpus:

- `gpt2`
- `xlm-roberta-base`

I computed:

- tokens per parallel sentence
- tokens per whitespace word
- tokens per grapheme cluster
- tokens per UTF-8 byte

### Result

| tokenizer        | lang | sentences | tok/sentence | tok/word | tok/grapheme | tok/UTF-8 byte |
| ---------------- | ---- | --------: | -----------: | -------: | -----------: | -------------: |
| gpt2             | eng  |       997 |       25.818 |    1.228 |       0.2056 |         0.2055 |
| gpt2             | hin  |       997 |      192.165 |    7.786 |       2.3250 |         0.5945 |
| gpt2             | kan  |       997 |      350.876 |   22.672 |       4.0594 |         0.9786 |
| gpt2             | tam  |       997 |      398.364 |   24.617 |       4.2043 |         0.9959 |
| gpt2             | tel  |       997 |      336.664 |   20.482 |       4.5625 |         0.9907 |
| xlm-roberta-base | eng  |       997 |       29.082 |    1.384 |       0.2316 |         0.2314 |
| xlm-roberta-base | hin  |       997 |       36.749 |    1.489 |       0.4446 |         0.1137 |
| xlm-roberta-base | kan  |       997 |       39.744 |    2.568 |       0.4598 |         0.1109 |
| xlm-roberta-base | tam  |       997 |       39.205 |    2.423 |       0.4138 |         0.0980 |
| xlm-roberta-base | tel  |       997 |       38.830 |    2.362 |       0.5262 |         0.1143 |

### Revision / conclusion

GPT-2 is extremely inefficient on Indic scripts, especially Dravidian languages. XLM-RoBERTa is much more balanced across English, Hindi, Kannada, Tamil, and Telugu. For routing and cost, tokens per parallel sentence is the main number I would use because it holds semantic content roughly constant across languages.

## 2026-07-26 - Day 4 - Final Evidence Strengthening and Reproducibility Scripts

### Goal

Strengthen the audit so the measured claims can be reproduced from the repository.

### Work completed

- Added `partA/prepare_flores_corpus.py` so the FLORES-200 corpus construction can be reproduced.
- Added `partA/analyze_tokenizers.py` so the corrected tokenizer table can be reproduced.
- Added measured effect tables for the claimed `fertility.py` flaws.
- Clarified the suspicious-looking CLI behavior that is actually correct: `--corpus` must be repeated because the script uses `argparse` with `action="append"`.
- Clarified Part B goodput using direct output-token goodput and a second consistency check from `reported_tok_s`.

### Reproducibility commands

```Python
python starter_kit/fertility.py --corpus eng=starter_kit/corpus_sample/eng_sample.txt --corpus hin=starter_kit/corpus_sample/hin_sample.txt --tokenizer gpt2
python -X utf8 partA/prepare_flores_corpus.py
python -X utf8 partA/analyze_tokenizers.py
```

## 2026-07-27 - Day 5 - Final Report Polish and Repository Review

### Planned work

- Re-read `partA/tokenizer_audit.md`, `partB/capacity_reconciliation.md`, and `partC/memo.md` for clarity.
- Check that every claimed flaw has measured evidence.
- Confirm that `README.md` explains the repository structure and reproduction commands.
- Confirm that `AI_USAGE.md` honestly describes how AI assistance was used.
- Review Git history for meaningful milestone commits.
- Run final local reproducibility checks before pushing.

### Expected result

The repository should be ready for GitHub submission after this review.
