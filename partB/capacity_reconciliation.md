# Part B - Capacity Reconciliation

## B1. KV-cache capacity

The KV cache stores both K and V for every token at every layer.

KV-cache memory per token:

```text
28 layers
* 2                 # K and V
* 8 KV heads
* 128 head_dim
* 2 bytes           # fp16
= 114,688 bytes/token
≈ 112 KiB/token
```

Usable GPU memory for serving:

```text
24 GB * 0.92 = 22.08 GB
22.08 GB - 1.6 GB non-KV overhead = 20.48 GB available for KV cache
```

KV-cache memory for one full 4096-token sequence:

```text
114,688 bytes/token * 4096 tokens
= 469,762,048 bytes
≈ 448 MiB per sequence
```

Maximum full 4096-token concurrent sequences:

```text
20.48 GB / 469,762,048 bytes
≈ 46 sequences
```

So the model should fit roughly **46 full 4096-token sequences** before KV-cache memory becomes the bottleneck.

The theoretical capacity is approximately 46 concurrent sequences, while the benchmark begins showing KV pressure around batch 32-48, consistent with runtime overheads and scheduler behavior.

## B2. Long-context throughput anomaly

In the long-context sweep, `prompt_len = 3584` and `gen_len = 512`, so each request reaches the full 4096-token context length.

Naively, throughput should improve as batch size increases. That happens up to batch 24, but then it breaks:

| batch_size | wall_clock_s | reported_tok_s | goodput_tok_s | preempted_seqs | kv_cache_util |
| ---------: | -----------: | -------------: | ------------: | -------------: | ------------: |
|         16 |        49.97 |         1311.4 |        163.94 |              0 |          0.62 |
|         24 |        61.16 |         1607.4 |        200.92 |              0 |          0.93 |
|         32 |        94.71 |         1384.0 |        172.99 |              7 |          0.97 |
|         48 |       151.41 |         1298.5 |        162.31 |             23 |          0.97 |

The anomaly is that throughput peaks at batch 24, then drops at batch 32 and batch 48 even though batch size increases.

The mechanism is KV-cache saturation. At batch 24, `kv_cache_util = 0.93` and `preempted_seqs = 0`, so the run still fits. At batch 32 and batch 48, `kv_cache_util = 0.97` and preemptions appear. The scheduler preempted 7 sequences at batch 32 and 23 sequences at batch 48. Those preemptions add overhead and reduce useful decode throughput.

A safer deployment config would cap long-context batch size around 24 for 4096-token requests. Based on the log, this would keep goodput around 200.92 output tok/s instead of dropping to 172.99 tok/s at batch 32 or 162.31 tok/s at batch 48.

## B3. Correcting the report's throughput interpretation

`REPORT_v0.md` misreads `reported_tok_s`. That column is the harness's built-in throughput counter, but for long prompts it appears to count total processed tokens, including prompt/prefill tokens, not only generated output tokens.

For the batch-24 long-prompt row:

```text
prompt_len = 3584
gen_len = 512
num_requests = 24
wall_clock_s = 61.16
reported_tok_s = 1607.4
```

Honest goodput should count only generated output tokens per second.

Method 1: derive directly from generated output tokens:

```text
num_requests * gen_len / wall_clock_s
= 24 * 512 / 61.16
= 12,288 / 61.16
= 200.92 output tokens/s
```

Method 2: infer the harness denominator from the log. If `reported_tok_s` counts total processed tokens (`prompt_len + gen_len`), then converting it to output-token throughput gives:

```text
reported_tok_s * (gen_len / (prompt_len + gen_len))
= 1607.4 * (512 / 4096)
= 1607.4 * 0.125
= 200.93 output tokens/s
```

This independently matches Method 1, which supports the interpretation that `reported_tok_s` is counting total processed tokens rather than only generated output tokens. The honest goodput is about **201 output tokens/s**, not **1607.4 output tokens/s**.

What the report should have said: batch 24 is the best long-context point in the log, but its useful generated-output throughput is about **201 output tokens/s**. The higher `reported_tok_s` number should be described as total processed token throughput, not user-visible output goodput.

## B4. Serving metric to confirm the mechanism

The single serving-stack metric I would check is the scheduler preemption counter, such as `preempted_seqs` or the serving system's equivalent preemption metric.

If the B2 mechanism is correct, this counter should stay at 0 while the long-context batch fits in KV cache, then become nonzero once KV cache is saturated. That is exactly what the log shows: batch 24 has `preempted_seqs = 0`, batch 32 has `preempted_seqs = 7`, and batch 48 has `preempted_seqs = 23`.
