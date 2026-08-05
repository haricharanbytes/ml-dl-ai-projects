import numpy as np

def scaled_dot_product_attention(q, k, v):
    head_dim = q.shape[-1]
    scores = np.einsum("hd,shd->hs", q, k) / np.sqrt(head_dim)
    scores = scores - scores.max(axis=-1, keepdims=True)  # numerical stability
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)
    out = np.einsum("hs,shd->hd", weights, v)
    return out


class ToyAttentionLayer:

    def __init__(self, num_heads, head_dim, embed_dim, rng):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.embed_dim = embed_dim
        scale = 1.0 / np.sqrt(embed_dim)
        self.Wq = rng.normal(0, scale, (embed_dim, num_heads, head_dim)).astype(np.float32)
        self.Wk = rng.normal(0, scale, (embed_dim, num_heads, head_dim)).astype(np.float32)
        self.Wv = rng.normal(0, scale, (embed_dim, num_heads, head_dim)).astype(np.float32)

    def project(self, x):
        """x: (embed_dim,) -> q, k, v each (num_heads, head_dim)"""
        q = np.einsum("e,ehd->hd", x, self.Wq)
        k = np.einsum("e,ehd->hd", x, self.Wk)
        v = np.einsum("e,ehd->hd", x, self.Wv)
        return q, k, v


class PagedKVCache:
    def __init__(self, num_layers, num_heads, head_dim, page_size, total_pages, seed=0):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.total_pages = total_pages

        self.K = [np.zeros((total_pages, page_size, num_heads, head_dim), dtype=np.float32)
                  for _ in range(num_layers)]
        self.V = [np.zeros((total_pages, page_size, num_heads, head_dim), dtype=np.float32)
                  for _ in range(num_layers)]

        self.free_pages = list(range(total_pages))
        self.ref_count = {}

    # allocation
    def allocate_page(self):
        if not self.free_pages:
            raise MemoryError("PagedKVCache OOM -- no free pages")
        pid = self.free_pages.pop(0)
        self.ref_count[pid] = 1
        return pid

    def share_page(self, pid):
        self.ref_count[pid] = self.ref_count.get(pid, 1) + 1

    def release_page(self, pid):
        self.ref_count[pid] -= 1
        if self.ref_count[pid] <= 0:
            self.free_pages.append(pid)
            del self.ref_count[pid]

    def cow_copy(self, pid, layer_range):
        new_pid = self.allocate_page()
        for layer in layer_range:
            self.K[layer][new_pid] = self.K[layer][pid].copy()
            self.V[layer][new_pid] = self.V[layer][pid].copy()
        self.release_page(pid)
        return new_pid

    @property
    def utilisation(self):
        used = self.total_pages - len(self.free_pages)
        return used / self.total_pages * 100

    @property
    def bytes_used(self):
        used = self.total_pages - len(self.free_pages)
        per_page = self.page_size * self.num_heads * self.head_dim * 4  # fp32
        return used * per_page * 2 * self.num_layers  # *2 for K and V

    # writing / reading 
    def write_token(self, layer, pid, slot, k_vec, v_vec):
        self.K[layer][pid, slot] = k_vec
        self.V[layer][pid, slot] = v_vec

    def gather(self, layer, block_table, num_tokens):
        k_chunks, v_chunks = [], []
        remaining = num_tokens
        for pid in block_table:
            take = min(self.page_size, remaining)
            k_chunks.append(self.K[layer][pid, :take])
            v_chunks.append(self.V[layer][pid, :take])
            remaining -= take
        return np.concatenate(k_chunks, axis=0), np.concatenate(v_chunks, axis=0)