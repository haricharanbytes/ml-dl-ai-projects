# PagedAttention, Visualized

An interactive, from-scratch simulation of [vLLM's PagedAttention](https://arxiv.org/abs/2309.06180)
(Kwon et al., 2023) — built to *prove*, not just illustrate, why paging the
KV cache saves memory without changing model outputs.

Unlike a purely conceptual demo, the paged cache here stores **real numpy
float32 K/V tensors**, produced by actual (untrained, fixed-weight)
multi-head attention layers. That means the dashboard can numerically
verify that attention computed over scattered physical pages is
bit-identical to attention computed over one contiguous array — paging is
a memory-layout optimization, not an approximation.

## Project structure

```
attention_core.py   Real scaled-dot-product attention (numpy) +
                     PagedKVCache: physical pages that hold actual tensors,
                     with allocate / share / release / copy-on-write.
simulator.py         Runs batches of requests through real attention,
                     writing into both a paged cache and a naive
                     contiguous cache, plus a correctness checker,
                     a CoW demo, and the naive-vs-paged utilisation
                     benchmark.
dashboard.py          Streamlit app tying it all together with live,
                     adjustable visuals.
```

## Run it

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

## What the dashboard shows

1. **Page grid** — every physical page in the pool, color-coded by owning
   request. Free a request live and watch its pages turn grey and become
   instantly reusable.
2. **Correctness proof** — attention run through the paged cache vs. a
   plain contiguous cache, same random query, diffed live. Diff is 0.0.
3. **Copy-on-Write** — N requests share one physical copy of a system
   prompt's real KV tensors via reference counting; one request diverges
   and gets its own physical copy while the rest keep sharing.
4. **Utilisation benchmark** — reproduces the paper's headline result:
   naive allocation utilises ~20–30% of reserved memory regardless of
   batch size; paged allocation stays ~95%+.

## What's simplified vs. real vLLM

- Attention layers use random fixed weights (untrained) — the point is
  memory layout and math correctness, not model quality.
- No actual GPU kernels, CUDA graphs, or PagedAttention's custom attention
  kernel — everything runs on CPU with numpy einsum.
- No scheduler, preemption, or block-level swapping to CPU/disk.
- Single-node, single-process — no distributed KV cache considerations.

## Possible next steps

- Swap numpy for torch tensors + `torch.cuda` to move real data onto a GPU
  and measure actual allocation/free latency.
- Add an eviction policy (LRU, priority) when the page pool is full and
  compare hit rates across policies.
- Add a toy scheduler that admits/preempts requests based on free-page
  count, visualized as a queue alongside the page grid.
- Wire in a tiny trained transformer (e.g. nanoGPT) so the dashboard
  produces real generated text alongside the memory visualization.