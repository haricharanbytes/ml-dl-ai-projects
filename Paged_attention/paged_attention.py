import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

random.seed(42)
np.random.seed(42)

NUM_LAYERS  = 32
NUM_HEADS   = 32
HEAD_DIM    = 128
BYTES_FP16  = 2
PAGE_SIZE   = 16   
MAX_SEQ_LEN = 2048

KV_BYTES_PER_TOKEN = 2 * NUM_LAYERS * NUM_HEADS * HEAD_DIM * BYTES_FP16
KV_MB_PER_TOKEN    = KV_BYTES_PER_TOKEN / 1024 / 1024

print("=" * 60)
print("SECTION 1 — Naive KV Cache: The Waste Problem")
print("=" * 60)

AVG_RESPONSE = 500   # realistic average tokens generated

pre_allocated_mb = MAX_SEQ_LEN  * KV_MB_PER_TOKEN
actually_used_mb = AVG_RESPONSE * KV_MB_PER_TOKEN

print(f"\nKV cache per token    : {KV_BYTES_PER_TOKEN:,} bytes")
print(f"Pre-allocated/request : {pre_allocated_mb:.2f} MB  ({MAX_SEQ_LEN} tokens)")
print(f"Actually used/request : {actually_used_mb:.2f} MB  ({AVG_RESPONSE} tokens)")
print(f"Utilisation           : {actually_used_mb / pre_allocated_mb * 100:.1f}%")
print(f"Wasted per request    : {pre_allocated_mb - actually_used_mb:.2f} MB")

NUM_USERS = 100
wasted_gb = (pre_allocated_mb - actually_used_mb) * NUM_USERS / 1024
print(f"\nAcross {NUM_USERS} concurrent users → {wasted_gb:.2f} GB wasted")
print("\n→ Naive systems utilise only 20–38% of allocated KV cache memory")
print("  (source: original Paged Attention vLLM paper)")

print("\n" + "=" * 60)
print("SECTION 2 — Paged Attention: Pages + Block Table")
print("=" * 60)

"""
Instead of one large contiguous block per request:
  - KV cache is split into fixed-size pages (PAGE_SIZE tokens each)
  - Pages are allocated on demand, can live anywhere in GPU memory
  - Each request keeps a block_table: logical index → physical page id
"""

class PagePool:
    def __init__(self, total_pages):
        self.free      = list(range(total_pages))
        self.total     = total_pages
        self.ref_count = defaultdict(int)

    def allocate(self):
        if not self.free:
            raise MemoryError("OOM — no free pages")
        pid = self.free.pop(0)
        self.ref_count[pid] = 1
        return pid

    def release(self, pid):
        self.ref_count[pid] -= 1
        if self.ref_count[pid] <= 0:
            self.free.append(pid)
            del self.ref_count[pid]

    def share(self, pid):
        """Increment ref count — another request is sharing this page."""
        self.ref_count[pid] += 1

    def cow_copy(self, pid):
        """CoW: allocate a new page, decrement ref on the old one."""
        new_pid = self.allocate()
        self.release(pid)
        return new_pid

    @property
    def utilisation(self):
        return (self.total - len(self.free)) / self.total * 100


class PagedRequest:
    def __init__(self, req_id, pool: PagePool):
        self.id          = req_id
        self.pool        = pool
        self.block_table = []   # logical index → physical page id
        self.tokens      = 0

    def generate_token(self):
        if self.tokens % PAGE_SIZE == 0:   # page boundary → allocate new page
            self.block_table.append(self.pool.allocate())
        self.tokens += 1

    def free(self):
        for pid in self.block_table:
            self.pool.release(pid)
        self.block_table.clear()


pool = PagePool(total_pages=512)
requests = [PagedRequest(f"req-{i}", pool) for i in range(5)]
token_counts = [320, 48, 160, 96, 272]

for req, n in zip(requests, token_counts):
    for _ in range(n):
        req.generate_token()

print("\nRequest state after generation:")
print(f"  {'ID':<10} {'Tokens':>8} {'Pages':>7} {'Last-page waste':>16}")
for req in requests:
    waste = req.tokens % PAGE_SIZE
    waste = PAGE_SIZE - waste if waste else 0
    print(f"  {req.id:<10} {req.tokens:>8} {len(req.block_table):>7} {waste:>16} tokens")

print(f"\nPool utilisation : {pool.utilisation:.1f}%")
requests[1].free()
print(f"After freeing req-1 → utilisation: {pool.utilisation:.1f}%  (pages immediately reusable)")

print("\n" + "=" * 60)
print("SECTION 3 — Copy-on-Write: Shared System Prompts")
print("=" * 60)

"""
If N requests share a system prompt, naive allocation stores N copies.
With CoW, all requests point to the SAME physical pages.
A private copy is made only when a request writes a diverging token.
"""

cow_pool    = PagePool(total_pages=512)
SYSTEM_TOKENS = 200
system_pages  = math.ceil(SYSTEM_TOKENS / PAGE_SIZE)
shared_pids   = [cow_pool.allocate() for _ in range(system_pages)]
print(f"\nSystem prompt → {system_pages} shared pages: {shared_pids}")

N = 10
user_tables = []
for i in range(N):
    table = list(shared_pids)
    for pid in shared_pids:
        cow_pool.share(pid)     # ref count up — no physical copy
    user_tables.append(table)

saved_mb = (system_pages * N - system_pages) * PAGE_SIZE * KV_MB_PER_TOKEN
print(f"\nStoring system prompt for {N} requests:")
print(f"  Naive : {system_pages * N} pages  ({system_pages * N * PAGE_SIZE * KV_MB_PER_TOKEN:.1f} MB)")
print(f"  CoW   : {system_pages} pages   ({system_pages * PAGE_SIZE * KV_MB_PER_TOKEN:.1f} MB)")
print(f"  Saved : {saved_mb:.1f} MB")

old_pid                 = user_tables[3][-1]
new_pid                 = cow_pool.cow_copy(old_pid)
user_tables[3][-1]      = new_pid
print(f"\nReq-3 diverges → CoW: old page {old_pid} → new page {new_pid}")
print(f"All other {N-1} requests still share page {old_pid} unaffected")

print("\n" + "=" * 60)
print("SECTION 4 -- Utilisation: Naive vs Paged")
print("=" * 60)

def naive_utilisation(n, max_seq=2048, avg=500, std=200):
    actual = np.clip(np.random.normal(avg, std, n).astype(int), 200, max_seq)
    return actual.sum() / (max_seq * n) * 100, actual

def paged_utilisation(actual_tokens, page_size=PAGE_SIZE):
    pages = np.ceil(actual_tokens / page_size).astype(int)
    return actual_tokens.sum() / (pages * page_size).sum() * 100

batch_sizes = [10, 25, 50, 100, 200]
naive_u, paged_u = [], []

print(f"\n  {'Batch':>6}   {'Naive':>8}   {'Paged':>8}")
for bs in batch_sizes:
    nu, actual = naive_utilisation(bs)
    pu = paged_utilisation(actual)
    naive_u.append(nu)
    paged_u.append(pu)
    print(f"  {bs:>6}   {nu:>7.1f}%   {pu:>7.1f}%")