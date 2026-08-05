import math
import numpy as np
from dataclasses import dataclass, field
from attention_core import ToyAttentionLayer, PagedKVCache, scaled_dot_product_attention


@dataclass
class RequestState:
    req_id: str
    num_tokens: int
    block_table: list = field(default_factory=list)   # logical page idx -> physical pid
    naive_K: list = None   # per-layer contiguous (num_tokens, num_heads, head_dim)
    naive_V: list = None


def build_layers(num_layers, num_heads, head_dim, embed_dim, seed=0):
    rng = np.random.default_rng(seed)
    return [ToyAttentionLayer(num_heads, head_dim, embed_dim, rng) for _ in range(num_layers)]


def run_batch(
    num_layers, num_heads, head_dim, embed_dim, page_size,
    total_pages, token_counts, seed=0,
):
    """
    Runs `len(token_counts)` requests through real attention, writing K/V
    into a PagedKVCache. Also keeps a naive contiguous copy per request for
    correctness verification. Returns everything the dashboard needs.
    """
    layers = build_layers(num_layers, num_heads, head_dim, embed_dim, seed)
    cache = PagedKVCache(num_layers, num_heads, head_dim, page_size, total_pages, seed)
    rng = np.random.default_rng(seed + 1)

    requests = [RequestState(f"req-{i}", n) for i, n in enumerate(token_counts)]
    for r in requests:
        r.naive_K = [np.zeros((r.num_tokens, num_heads, head_dim), dtype=np.float32) for _ in range(num_layers)]
        r.naive_V = [np.zeros((r.num_tokens, num_heads, head_dim), dtype=np.float32) for _ in range(num_layers)]

    events = []  # log of (step, req_id, event_type, detail) for the dashboard timeline

    for r in requests:
        for t in range(r.num_tokens):
            # page-boundary allocation (identical logic to the original PagePool)
            if t % page_size == 0:
                pid = cache.allocate_page()
                r.block_table.append(pid)
                events.append({"req": r.req_id, "event": "alloc_page", "page": pid, "token": t})
            pid = r.block_table[-1]
            slot = t % page_size

            # fabricate an input embedding for this token (deterministic per req/token)
            x = rng.normal(0, 1, embed_dim).astype(np.float32)

            for layer_idx, layer in enumerate(layers):
                q, k, v = layer.project(x)
                cache.write_token(layer_idx, pid, slot, k, v)
                r.naive_K[layer_idx][t] = k
                r.naive_V[layer_idx][t] = v

    return layers, cache, requests, events


def verify_correctness(layers, cache, requests):
    """
    For each request, gather its paged K/V, run attention using the LAST
    token's query against the full paged history, and compare to the same
    attention run against the naive contiguous cache. Returns max abs diff
    per request per layer (should be ~0, i.e. float32 rounding only).
    """
    results = []
    rng = np.random.default_rng(999)
    for r in requests:
        row = {"req_id": r.req_id, "layer_diffs": []}
        # synth a query vector for the last position, same across both paths
        for layer_idx, layer in enumerate(layers):
            q = rng.normal(0, 1, (layer.num_heads, layer.head_dim)).astype(np.float32)

            paged_k, paged_v = cache.gather(layer_idx, r.block_table, r.num_tokens)
            out_paged = scaled_dot_product_attention(q, paged_k, paged_v)
            out_naive = scaled_dot_product_attention(q, r.naive_K[layer_idx], r.naive_V[layer_idx])

            diff = float(np.max(np.abs(out_paged - out_naive)))
            row["layer_diffs"].append(diff)
        row["max_diff"] = max(row["layer_diffs"])
        results.append(row)
    return results


def cow_demo(num_layers, num_heads, head_dim, embed_dim, page_size, total_pages,
             system_tokens, num_users, seed=0):
    """
    Simulates N requests sharing a real system-prompt KV cache via CoW,
    then diverging one request and physically copying its pages.
    """
    layers = build_layers(num_layers, num_heads, head_dim, embed_dim, seed)
    cache = PagedKVCache(num_layers, num_heads, head_dim, page_size, total_pages, seed)
    rng = np.random.default_rng(seed + 7)

    system_pages_needed = math.ceil(system_tokens / page_size)
    shared_pids = []
    for p in range(system_pages_needed):
        pid = cache.allocate_page()
        shared_pids.append(pid)
        for slot in range(min(page_size, system_tokens - p * page_size)):
            x = rng.normal(0, 1, embed_dim).astype(np.float32)
            for layer_idx, layer in enumerate(layers):
                q, k, v = layer.project(x)
                cache.write_token(layer_idx, pid, slot, k, v)

    user_tables = []
    for _ in range(num_users):
        table = list(shared_pids)
        for pid in shared_pids:
            cache.share_page(pid)
        user_tables.append(table)

    ref_counts_before = {pid: cache.ref_count[pid] for pid in shared_pids}

    # one request diverges: CoW-copy its last shared page
    diverge_idx = num_users // 2
    old_pid = user_tables[diverge_idx][-1]
    new_pid = cache.cow_copy(old_pid, layer_range=range(num_layers))
    user_tables[diverge_idx][-1] = new_pid

    ref_counts_after = {pid: cache.ref_count.get(pid, 0) for pid in shared_pids}

    naive_bytes = system_pages_needed * num_users * page_size * num_heads * head_dim * 4 * 2 * num_layers
    cow_bytes = system_pages_needed * page_size * num_heads * head_dim * 4 * 2 * num_layers

    return {
        "shared_pids": shared_pids,
        "system_pages_needed": system_pages_needed,
        "ref_counts_before": ref_counts_before,
        "ref_counts_after": ref_counts_after,
        "old_pid": old_pid,
        "new_pid": new_pid,
        "naive_bytes": naive_bytes,
        "cow_bytes": cow_bytes,
        "saved_bytes": naive_bytes - cow_bytes,
    }


def utilisation_benchmark(batch_sizes, max_seq=2048, avg=500, std=200, page_size=16, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for bs in batch_sizes:
        actual = np.clip(rng.normal(avg, std, bs).astype(int), 200, max_seq)
        naive_u = actual.sum() / (max_seq * bs) * 100
        pages = np.ceil(actual / page_size).astype(int)
        paged_u = actual.sum() / (pages * page_size).sum() * 100
        rows.append({"batch_size": bs, "naive_pct": naive_u, "paged_pct": paged_u})
    return rows