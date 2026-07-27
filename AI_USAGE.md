# AI Usage

I used AI assistance during this assignment as a structured coding and reasoning assistant. I did not treat AI output as the source of truth; I used it to move faster, then verified the important claims myself through commands, scripts, terminal outputs, and the provided starter files.

## Where AI helped

AI helped me with:

- Understanding the assignment requirements and breaking them into a checklist.
- Planning the repository structure and Git workflow.
- Setting up the local Python, Jupyter, tokenizer, and ML package environment.
- Inspecting the starter files and identifying what evidence was needed.
- Drafting and refining commands for tokenizer analysis and serving-capacity calculations.
- Drafting explanations in `NOTEBOOK.md`, `partA/tokenizer_audit.md`, `partB/capacity_reconciliation.md`, and `partC/memo.md`.
- Improving the reports so that claims included commands, measured numbers, and conclusions.

## Where AI misled me

One AI-assisted suggestion initially treated the failed command below as if the script could accept multiple `LANG=PATH` values after one `--corpus` flag:

```Python
python starter_kit/fertility.py --corpus eng=starter_kit/corpus_sample/eng_sample.txt hin=starter_kit/corpus_sample/hin_sample.txt
```

When I tested it, the script rejected the second corpus argument:

```Python
fertility.py: error: unrecognized arguments: hin=starter_kit/corpus_sample/hin_sample.txt
```

I then checked `python starter_kit/fertility.py -h` and the script's `argparse` setup. The correct usage is to repeat the `--corpus` flag once per corpus:

```Python
python starter_kit/fertility.py --corpus eng=starter_kit/corpus_sample/eng_sample.txt --corpus hin=starter_kit/corpus_sample/hin_sample.txt
```

So this was not a bug in the starter script. It was a command-usage mistake that I corrected through testing.

AI also helped draft early explanations, but I revised them where they were too confident without enough measured evidence. For example, I only kept claimed flaws after adding measured before/after numbers or an explicit note that the measured effect on the provided sample was zero.

## What I verified myself

I personally verified the numerical and technical claims by:

* Running the original `fertility.py` command on the starter corpus.
* Checking the `fertility.py` command-line interface with `-h`.
* Comparing line-average fertility against aggregate fertility.
* Testing `split(" ")` versus `split()` behavior.
* Building the FLORES-200 corpus files used in Part A.
* Running the corrected tokenizer analysis for `gpt2` and `xlm-roberta-base`.
* Computing KV-cache capacity from `model_spec.md`.
* Computing honest long-context goodput from `bench_log.csv`.
* Checking that the reported conclusions matched the command outputs.

The final responsibility for the conclusions, numbers, and recommendations is mine.
