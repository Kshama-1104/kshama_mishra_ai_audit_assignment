# Part C - Decision Memo

## Recommendation

I recommend starting with prompt engineering plus a small evaluation set, and only moving to synthetic SFT if the prompt-only approach fails the kill criterion. I would not choose a separate inference-time rewriter as the first path because it adds serving latency, another model to monitor, and more failure modes close to launch.

## Assumptions

- The main problem is style: replies are understandable but too formal or textbook-like.
- The base assistant already supports Hindi, Kannada, Tamil, Telugu, Bengali, and Marathi.
- We have one A100-80GB for 2 weeks, but launch review is in 3 weeks, so the first solution must be fast to test.
- Native review is limited to Hindi and Kannada for 10 hours/week, so we need a small but high-signal evaluation set.
- No external API budget means synthetic data must come from local/open models or internal generation, not paid APIs.

## Back-of-envelope arithmetic

I would build a 300-prompt style evaluation set first: 50 prompts per language across the 6 target languages. For Hindi and Kannada, the native reviewer can directly score outputs. If one review takes about 2 minutes, then 100 Hindi/Kannada examples take about 200 minutes, or 3.3 hours per pass. That fits within the 10 hours/week reviewer budget and leaves time for a second pass.

For prompt engineering, serving cost is almost zero beyond slightly longer prompts. If the style instruction adds about 100 tokens to the system/developer prompt, that cost is paid once per request and can be tested immediately.

For SFT, even a small dataset of 5,000 to 10,000 synthetic casualized pairs would need generation, filtering, training, and review. A LoRA-style SFT on one A100-80GB is feasible in the 2-week compute window, but the risky part is not training time; it is data quality and multilingual review coverage.

For a rewriter model under 1B parameters, serving would add one extra inference call per assistant response. Even if the model is small, it increases latency and can distort factual content, so I would keep it as a fallback, not the first launch path.

## Success metric

On the held-out 300-prompt evaluation set, the chosen approach should achieve:

- At least 80% of Hindi and Kannada reviewed outputs rated "casual and natural" by the native reviewer.
- No more than 5% of reviewed outputs rated as meaningfully worse in correctness or helpfulness.
- For the remaining four languages, automatic and manual spot checks should show no obvious script, language-mixing, or over-formality failures in at least 40 examples per language.

## Kill criterion

If by the end of day 3 prompt engineering cannot reach at least 70% "casual and natural" on the first Hindi/Kannada review pass, or if it causes more than 5% correctness/helpfulness regressions, I would abandon prompt-only as the launch path and start synthetic SFT immediately.

## First-day experiment

On day 1, I would create 50 Hindi and 50 Kannada prompts from real assistant use cases, run the current model with the existing prompt and with two casual-style prompt variants, then ask the native reviewer to blind-rate the outputs for naturalness, correctness, and preference. This gives a fast go/no-go signal before spending time on SFT data generation or adding a rewriter model.
