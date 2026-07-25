# Part A - Tokenizer Audit

## A1. Evaluation corpus

I built the evaluation corpus from the FLORES-200 dev split using the open Hugging Face mirror `Muennighoff/flores200`. I initially attempted to use `facebook/flores`, but that dataset was gated in my environment and required authentication, so I used the open mirror instead.

The corpus has 997 aligned sentences per language:

| file                               | language | FLORES code  | lines |
| ---------------------------------- | -------- | ------------ | ----: |
| `partA/corpus_flores200/eng.txt` | English  | `eng_Latn` |   997 |
| `partA/corpus_flores200/hin.txt` | Hindi    | `hin_Deva` |   997 |
| `partA/corpus_flores200/kan.txt` | Kannada  | `kan_Knda` |   997 |
| `partA/corpus_flores200/tam.txt` | Tamil    | `tam_Taml` |   997 |
| `partA/corpus_flores200/tel.txt` | Telugu   | `tel_Telu` |   997 |

Domain: FLORES-200 is a multilingual benchmark corpus built from Wikimedia-style text. The sample rows include news/wiki-style domains such as Wikinews health articles.

Preprocessing:

- Loaded the `dev` split.
- Kept non-empty `sentence` fields.
- Wrote one sentence per line as UTF-8 text.
- Preserved original casing and punctuation for the corpus files.

Limitations: this corpus is much better than the 10-line starter sample, but it still cannot represent every production input. It is mostly written benchmark/news/wiki-style text, not chat logs, code-mixed user queries, or informal assistant conversations. It also has only 997 sentences per language, so it is useful for a controlled audit but not a final production tokenizer evaluation.

## A2. fertility.py script and metric audit

### Smoke-test reproduction

Command:

```Python
python starter_kit/fertility.py --corpus eng=starter_kit/corpus_sample/eng_sample.txt --corpus hin=starter_kit/corpus_sample/hin_sample.txt --tokenizer gpt2
```

Output:

| language | original fertility tok/word | original tok/char |
| -------- | --------------------------: | ----------------: |
| eng      |                        1.27 |             0.226 |
| hin      |                        7.45 |             1.579 |

The starter script reports Hindi as `5.89x` the fertility of English.

### Flaw 1: line-level averaging distorts the metric

The script computes fertility by averaging per-line ratios:

```python
per_line_fertility.append(len(tokens) / len(words))
...
return sum(per_line_fertility) / n
```

A better aggregate metric is:

```text
sum(tokens) / sum(words)
```

Measured evidence on the starter sample:

| language | original line-avg tok/word | aggregate tok/word |  delta |
| -------- | -------------------------: | -----------------: | -----: |
| eng      |                     1.2652 |             1.2692 | +0.32% |
| hin      |                     7.4485 |             7.5246 | +1.02% |

Direction of distortion: on this sample, line-level averaging slightly underestimates both English and Hindi fertility.

### Flaw 2: `split(" ")` miscounts words

The script uses:

```python
words = line.split(" ")
```

This only splits on literal spaces. It creates empty words for repeated spaces and fails to split tabs or other whitespace.

Synthetic evidence isolating the bug:

| input                          | `split(" ")` count | `split()` count | problem                           |
| ------------------------------ | -------------------: | ----------------: | --------------------------------- |
| `hello world`                |                    2 |                 2 | no issue                          |
| `hello  world`               |                    3 |                 2 | repeated spaces create empty word |
| `hello\tworld`               |                    1 |                 2 | tab is not counted as separator   |
| `नमस्ते  दुनिया` |                    3 |                 2 | repeated spaces create empty word |

Measured effect on the provided starter corpus:

| language | original`split(" ")` tok/word | after`split()` tok/word | change |
| -------- | ------------------------------: | ------------------------: | -----: |
| eng      |                          1.2652 |                    1.2652 |  0.00% |
| hin      |                          7.4485 |                    7.4485 |  0.00% |

Direction and magnitude on the provided corpus: no measured change, because the starter sample does not contain repeated spaces or tab-separated words. The bug is still real because it would distort results on messier corpora.

Direction of possible distortion: repeated spaces inflate the word count, which lowers `tokens / word`; tabs reduce the word count, which raises `tokens / word`.

Fix:

```python
words = line.split()
```

### Flaw 3: `tok/word` is the wrong main metric for routing and cost

Even if the script computed `tokens / whitespace word` perfectly, it is not the right headline metric for cross-language routing decisions. Whitespace words are not stable units across languages. Languages differ in morphology, script, and spacing conventions, so "per word" does not hold content constant.

For routing and cost, the real question is:

```text
How many tokens does the same user intent/content consume in each language?
```

That is why my corrected analysis in A3 uses `tokens per parallel sentence` as the headline metric, with `tokens per grapheme` and `tokens per UTF-8 byte` as supporting diagnostics.

Measured effect of denominator choice on the corrected FLORES corpus:

| comparison                                               |  value |
| -------------------------------------------------------- | -----: |
| GPT-2 Tamil vs English using`tokens/word`              | 20.05x |
| GPT-2 Tamil vs English using`tokens/parallel sentence` | 15.43x |

Direction and magnitude of distortion: using `tokens/word` makes Tamil look about `20.05x` English, while the parallel-sentence metric makes it about `15.43x` English. That is a large difference in the routing story. `tokens/parallel sentence` is better for serving and routing because each row is aligned content with roughly the same meaning, so it compares token cost for equivalent content rather than for language-specific whitespace words.

### Suspicious-looking thing that is actually correct

At first, the command-line interface looks suspicious because this fails:

```Python
python starter_kit/fertility.py --corpus eng=starter_kit/corpus_sample/eng_sample.txt hin=starter_kit/corpus_sample/hin_sample.txt
```

The error is:

```Python
fertility.py: error: unrecognized arguments: hin=starter_kit/corpus_sample/hin_sample.txt
```

But this is not a bug. The script defines `--corpus` with `action="append"`, so each corpus must repeat the flag:

```Python
python starter_kit/fertility.py --corpus eng=starter_kit/corpus_sample/eng_sample.txt --corpus hin=starter_kit/corpus_sample/hin_sample.txt
```

This interface is explicit and works correctly.

## A3. Corrected tokenizer analysis

I ran the corrected analysis on the FLORES-style parallel corpus using two tokenizers:

- `gpt2`
- `xlm-roberta-base`

I used multiple denominators because `tok/word` alone is not reliable across languages:

- tokens per parallel sentence
- tokens per word
- tokens per grapheme
- tokens per UTF-8 byte

| tokenizer        | language | sentences | tokens/sentence | tokens/word | tokens/grapheme | tokens/UTF-8 byte |
| ---------------- | -------- | --------: | --------------: | ----------: | --------------: | ----------------: |
| gpt2             | eng      |       997 |          25.818 |       1.228 |          0.2056 |            0.2055 |
| gpt2             | hin      |       997 |         192.165 |       7.786 |          2.3250 |            0.5945 |
| gpt2             | kan      |       997 |         350.876 |      22.672 |          4.0594 |            0.9786 |
| gpt2             | tam      |       997 |         398.364 |      24.617 |          4.2043 |            0.9959 |
| gpt2             | tel      |       997 |         336.664 |      20.482 |          4.5625 |            0.9907 |
| xlm-roberta-base | eng      |       997 |          29.082 |       1.384 |          0.2316 |            0.2314 |
| xlm-roberta-base | hin      |       997 |          36.749 |       1.489 |          0.4446 |            0.1137 |
| xlm-roberta-base | kan      |       997 |          39.744 |       2.568 |          0.4598 |            0.1109 |
| xlm-roberta-base | tam      |       997 |          39.205 |       2.423 |          0.4138 |            0.0980 |
| xlm-roberta-base | tel      |       997 |          38.830 |       2.362 |          0.5262 |            0.1143 |

Using tokens per parallel sentence as the main metric, GPT-2 is much worse for Indic languages than English:

| language | GPT-2 tokens/sentence | ratio vs English |
| -------- | --------------------: | ---------------: |
| Hindi    |               192.165 |            7.44x |
| Kannada  |               350.876 |           13.59x |
| Tamil    |               398.364 |           15.43x |
| Telugu   |               336.664 |           13.04x |

The multilingual tokenizer `xlm-roberta-base` is much more balanced. It uses about 29.082 tokens/sentence for English and about 36.749 to 39.744 tokens/sentence for the Indic languages.

Conclusion: The original report's warning about GPT-2 token inflation is directionally correct, but the report uses weak evidence. The corrected analysis should use a larger parallel corpus and should make `tokens per parallel sentence` or production `tokens per request` the headline metric, not `tok/word`.

Tokenizer choice: I used `gpt2` because it matches the starter script and reproduces the original report's tokenizer path. I used `xlm-roberta-base` as a multilingual comparison tokenizer to test whether the observed Indic-language inflation is a general property or mainly a GPT-2-tokenizer issue. I did not include Llama-3 or IndicBERT because the assignment requires at least two tokenizers, and the goal here was to compare the starter tokenizer against one widely used multilingual tokenizer with reproducible local tooling.

`Tokens per parallel sentence` is not perfect, because translations can differ in length and style. However, it is still a better routing proxy than whitespace `tokens/word` because each row is intended to express the same underlying meaning across languages. For serving cost, the important question is how many tokens are needed for equivalent user content, not how many tokens are needed per language-specific whitespace word.

## A4. Recommendation memo

The original report's headline claim overstates the case because it depends on a tiny 10-line corpus and on `tok/word`, which is not a stable cross-language denominator. My corrected analysis uses a 997-sentence FLORES-style parallel corpus across English, Hindi, Kannada, Tamil, and Telugu.

Headline corrected numbers using `tokens per parallel sentence`:

| tokenizer        |    eng |     hin |     kan |     tam |     tel |
| ---------------- | -----: | ------: | ------: | ------: | ------: |
| gpt2             | 25.818 | 192.165 | 350.876 | 398.364 | 336.664 |
| xlm-roberta-base | 29.082 |  36.749 |  39.744 |  39.205 |  38.830 |

Recommendation: do not route or price Indic-language traffic based on the original GPT-2 fertility report. If the serving stack uses a GPT-2-like tokenizer, Indic-script requests are much more expensive in token budget, especially Kannada, Tamil, and Telugu. If possible, use a multilingual/Indic-aware tokenizer or model path for these languages. For routing and capacity planning, use `tokens per parallel sentence` or production `tokens per request` by language, not `tok/word`.

Biggest caveat: FLORES is a controlled written benchmark corpus, mostly wiki/news-style text. It is not a production chat corpus and does not capture code-mixing, informal spelling, transliteration, or short conversational queries.

Production metric to monitor: median and p95 input tokens per request by detected language, plus output tokens per request. This would catch the analysis being wrong if real traffic has very different token usage than the FLORES benchmark.
