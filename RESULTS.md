# Lab 17 Results

## Standard Benchmark

| Agent | Agent tokens | Prompt tokens | Recall | Quality | Memory bytes | Compactions |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 1428 | 14144 | 0.000 | 0.000 | 0 | 0 |
| Advanced | 3463 | 23244 | 0.976 | 0.976 | 3537 | 17 |

Advanced retains user facts across new threads through User.md. It costs more
prompt processing for ordinary short conversations because it loads the profile
and memory context on each turn.

## Long-Context Stress Benchmark

| Agent | Agent tokens | Prompt tokens | Recall | Quality | Memory bytes | Compactions |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 264 | 22257 | 0.000 | 0.000 | 0 | 0 |
| Advanced | 746 | 8951 | 1.000 | 1.000 | 259 | 26 |

Compact memory reduces the Advanced prompt load by about 60% in the long
conversation while persistent profile memory retains cross-session facts.

## Trade-offs and Risks

- Baseline is cheaper to reason about but deliberately forgets facts in a new thread.
- Advanced improves recall, but User.md grows over time and can preserve a wrong fact.
- A production version should add confidence checks and review/correction rules before
  writing untrusted user statements into persistent memory.
