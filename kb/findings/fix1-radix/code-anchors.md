# Code anchors: fix-1 forward `mamba_last_track_seqlen` across chunk transitions

Verified READ-ONLY on branch `vektory79`, HEAD `ebf071b` ("Memory"), working tree clean under
`python/freetoken/`. `git diff 1a444e4 --stat` over all 7 touched files is EMPTY: every file
below is byte-identical to commit 1a444e4, so all drift is relative to the TASK.md numbers only.

All snippets are verbatim (indentation preserved).

## 1. Finish-frozen donate - scheduler/cache.py `_cache_req_hybrid`

Anchor said ~346-362. Actual: L read at 348, guard 349-354, insert 357-358, block end 362.
Drift: +2 lines at the start of the block (347 vs ~345).

`python/freetoken/scheduler/cache.py:340-362` (comment block 340-346 documents WHY the
freeze-first donation ordering and dedup-free floor are correct by design; anchor said
~343-350 -> actual comment is 340-346, drift -3):

```
340            # A pending freeze (the tool-call anchor, or a prefill ×64 track the request
341            # finished too early to chunk-commit) is a strictly shorter prefix than the live
342            # donate below: insert it first and advance the dedup-free floor to its boundary
343            # -- [prefix_len, L) is now tree-owned by the donated node, so only [old, prefix_len)
344            # is this request's dup to free. The frozen slot is consumed either way (taken by
345            # the tree or freed here) and both ping-pong refs are dropped before
346            # _free_req_slots so nothing double-frees.
347            free_upto = old_handle.cached_len
348            L = req.mamba_last_track_seqlen
349            if (
350                L is not None
351                and 0 < L <= req.cached_len
352                and align_down(L, self.page_size) == L
353                and req.mamba_ping_pong is not None
354            ):
355                frozen_idx = 1 - req.mamba_next_track_idx
356                frozen = req.mamba_ping_pong[frozen_idx]
357                prefix_len, mamba_exist = self.prefix_cache.insert(
358                    req.input_ids[:L], page_indices[:L], frozen)
359                pool.free([s for s in req.mamba_ping_pong if mamba_exist or s != frozen])
360                req.mamba_ping_pong = None
361                self._free(page_indices[free_upto : max(free_upto, prefix_len)])
362                free_upto = max(free_upto, L)
```

Second design comment (live-donate page alignment), `cache.py:363-367`:

```
363            # Donate the live slot (final full-sequence state). The live state is at cached_len;
364            # only attach it when cached_len is itself the page-aligned node boundary (always for
365            # page_size==1). For page_size>1 a non-aligned cached_len would attach an over-advanced
366            # state to a shorter prefix node -> skip the finish-donate (the ×64 prefill snapshots
367            # remain as reuse points).
```

Note: the finish path does NOT reset `mamba_last_track_seqlen` after the frozen donate (only
`mamba_ping_pong = None` at 360) - safe because the req is being freed on this path.

## 2. Tool-call anchor consumer - scheduler/cache.py `snapshot_toolcall_anchor`

Anchor ~148-173. Actual 148-173. Drift: NONE.

```
148    def snapshot_toolcall_anchor(self, reqs: List[Req]) -> None:
149        """Freeze each decoding request's GDN state at its tool-call anchor, into the ping-pong
150        slot that is idle during decode (the kernel-side ×CHUNK track only runs on prefill
151        extends). Must run on the engine stream before the current step's kernels: cached_len
152        equals the anchor exactly when every enqueued step up to the anchor-consuming one has
153        been issued and the next (current) one has not, so the copy lands between them in
154        stream order. Reuses ``mamba_last_track_seqlen`` as the pending-donate mark -- the
155        prefill track's own pending freeze was consumed by the prefill-commit ``cache_req``
156        before any decode drain could set an anchor."""
157        if not self.is_hybrid:
158            return
159        pool = self.linear_state_pool
160        for r in reqs:
161            a = r.toolcall_anchor_len
162            if (
163                a is None
164                or r.mamba_ping_pong is None
165                or r.mamba_last_track_seqlen is not None
166                or r.cached_len != a
167                or align_down(a, self.page_size) != a
168            ):
169                continue
170            dst = r.mamba_ping_pong[r.mamba_next_track_idx]
171            pool.copy_from(r.linear_slot_idx, dst)
172            r.mamba_last_track_seqlen = a
173            r.mamba_next_track_idx = 1 - r.mamba_next_track_idx
```

Semantics: consume-if-None - it only freezes an anchor when L is None (guard at 165), then
CLAIMS L (172). This is the one consumer sensitive to L leaking into decode.

## 3. Ping-pong replacement alloc + locked-handle eviction race - scheduler/cache.py

Anchor ~398-401. Actual: race comment 398-400, `match_prefix` re-lock 401, replacement alloc
409-413. Drift: NONE. Context only, out of scope (chunk-commit path, `finished=False` branch):

```
394        prefix_len, mamba_exist = self.prefix_cache.insert(
395            req.input_ids[:L], page_indices[:L], frozen)
396        self.unlock(old_handle)
397        self._free(page_indices[old_handle.cached_len : prefix_len])
398        # Lock the committed snapshot node FIRST: the replacement-slot alloc below can trigger
399        # evict_mamba (via ensure_mamba_slots), which would otherwise reclaim this still-unlocked
400        # just-donated node -- freeing its KV pages under the still-decoding request.
401        m = self.prefix_cache.match_prefix(req.input_ids[:L])
...
409        if not mamba_exist:                                # tree took `frozen`; replace it
410            self.ensure_mamba_slots(1)
411            pp = list(req.mamba_ping_pong)
412            pp[frozen_idx] = pool.alloc(1)[0]
413            req.mamba_ping_pong = tuple(pp)
414        req.mamba_last_track_seqlen = None
```

Full chunk-commit donate block is 382-414: `L` read 383, `if L is None: return` 384-385,
misaligned skip that CLEARS L 386-391, frozen slot 392-393, insert 394-395, L cleared 414.

## 4. try_add_one forwarding - scheduler/prefill.py

Anchor ~236-250. Actual: `try_add_one` spans 230-268; the continuation branch is 234-245;
forwarded fields at 240-244. Drift: -2. CONFIRMED: `mamba_last_track_seqlen` is NOT forwarded.

`python/freetoken/scheduler/prefill.py:230-245`:

```
230    def try_add_one(self, pending_req: PendingReq) -> Req | None:
231        if self.token_budget <= 0:
232            return None
233
234        if chunked_req := pending_req.chunked_req:
235            return self._add_one_req(
236                pending_req=pending_req,
237                cache_handle=chunked_req.cache_handle,
238                table_idx=chunked_req.table_idx,
239                cached_len=chunked_req.cached_len,
240                linear_slot_idx=chunked_req.linear_slot_idx,
241                ping_pong=chunked_req.mamba_ping_pong,
242                next_track_idx=chunked_req.mamba_next_track_idx,
243                restore_src=None,  # continuation chunk already has live state
244                swa_evicted_seqlen=chunked_req.swa_evicted_seqlen,  # extend-free watermark so far
245            )
```

Fresh-admit branch 247-268 passes `restore_src=restore_src` (tree snapshot slot, `mr.mamba_value`)
and must keep L unset (default None) - correct as is.

`_add_one_req` signature 134-145 (params: `pending_req, cache_handle, table_idx, cached_len,
linear_slot_idx=None, ping_pong=None, next_track_idx=0, restore_src=None,
swa_evicted_seqlen=0`), then the Req construction:

```
196        is_chunked = chunk_size < remain_len
197        CLS = ChunkedReq if is_chunked else Req
...
209        req = CLS(
210            input_ids=pending_req.input_ids[: cached_len + chunk_size],
211            table_idx=table_idx,
212            cached_len=cached_len,
213            output_len=pending_req.output_len,
214            uid=pending_req.uid,
215            cache_handle=cache_handle,
216            sampling_params=pending_req.sampling_params,
217        )
218        req.mm_items = pending_req.mm_items
219        req.mrope_positions_full = pending_req.mrope_positions_full
220        req.mrope_delta = pending_req.mrope_delta
221        # Hybrid GDN per-request state slots (None for non-hybrid). On a fresh admit these are
222        # freshly allocated; on a chunked continuation they are inherited from the prior chunk.
223        req.linear_slot_idx = linear_slot_idx
224        req.mamba_ping_pong = ping_pong
225        req.mamba_next_track_idx = next_track_idx
226        req.mamba_restore_src = restore_src
227        req.swa_evicted_seqlen = swa_evicted_seqlen  # carry the extend-free watermark across chunks
228        return req
```

## 5. Continuation-Req construction + ChunkedReq storage + attach point

- The ONLY construction site of continuation Reqs in the whole package is the `req = CLS(...)`
  call above (prefill.py:209-217), with the class picked at 197. `find_usages`:
  `_add_one_req` is called only from `try_add_one` (prefill.py:235 continuation, 249 fresh);
  `try_add_one` is called only from `PrefillManager.schedule_next_batch` (prefill.py:315) and
  tests (`tests/scheduler/test_hybrid_cache_manager.py:133,138,153`).
- `ChunkedReq` (prefill.py:31-40) subclasses `Req` with no `__init__`; `Req` is
  `@dataclass(eq=False)` (python/freetoken/core.py:37) whose field
  `mamba_last_track_seqlen: int | None = None` (core.py:52) is therefore inherited and
  get/set by plain attribute. A `ChunkedReq` that tracked a boundary during its forward holds
  a non-None L until it is either committed or forwarded.
- `PendingReq.chunked_req` field: `python/freetoken/scheduler/utils.py:19`
  (`chunked_req: ChunkedReq | None = None`, dataclass at utils.py:14-36).
- Attach/detach: `python/freetoken/scheduler/prefill.py:313-319` in
  `PrefillManager.schedule_next_batch`:

```
313        for pending_req in self.pending_list:
314            is_continuation = pending_req.chunked_req is not None
315            if req := adder.try_add_one(pending_req):
316                pending_req.chunked_req = None
317                if isinstance(req, ChunkedReq):
318                    pending_req.chunked_req = req
319                    chunked_list.append(pending_req)
```

(Detach at 316, re-attach of the fresh continuation ChunkedReq at 318.)

## 6. Req field default - python/freetoken/core.py

Anchor "core.py ~52". Actual `python/freetoken/core.py:52` (NOT scheduler/core.py - that file
does not exist). Drift: NONE.

```
37  @dataclass(eq=False)
38  class Req:
...
52      mamba_last_track_seqlen: int | None = None      # chunk-aligned committed len of the last snapshot
```

## 7. Final-batch commit - scheduler/scheduler.py

`python/freetoken/scheduler.py` does not exist; the scheduler package is
`python/freetoken/scheduler/`. The anchor line is EXACT:
`python/freetoken/scheduler/scheduler.py:398` = `self.cache_manager.cache_req(req, finished=False)`.

Context, scheduler.py:384-398 (`_process_last_data`, non-finished prefill commit branch):

```
384                # NOTE: overlap scheduling may make the request freed twice, skip second free
385                if finished and req not in self.finished_reqs:
386                    self.decode_manager.remove_req(req)
387                    self._free_req_resources(req)
388                    new_finished_reqs.add(req)
389                elif batch.is_prefill and req.table_idx != -1:
390                    # for prefill, non-chunk req, cache the prefix.
391                    # Polymorphic: the DSV4 naive manager keeps the request's slots (no-op);
392                    # the generic manager inserts the prefix into its radix/naive cache.
393                    # table_idx == -1 is defense-in-depth: aborts mark in-flight requests
394                    # instead of freeing them (handled above), so a freed request should
395                    # never reach this commit -- but if a future path frees one early, skip
396                    # rather than re-read the freed page-table row (and on hybrid, deref the
397                    # None'd GDN ping-pong slots).
398                    self.cache_manager.cache_req(req, finished=False)
```

KEY GATE (confirms the TASK.md premise "only the batch that completes prefill produces a
commit"), scheduler.py:322-331 - INTERMEDIATE chunk reqs are skipped before the commit:

```
322                if isinstance(req, ChunkedReq):
323                    # Don't cache intermediate chunks; the full prompt is cached once when the
324                    # final chunk is processed. Caching here snapshots a handle the next chunk
325                    # already copied (overlap), so cache_req double-frees the prior chunk.
326                    if req.aborted:
327                        # Aborted mid-chunked-prefill while this chunk was in flight: the abort
328                        # popped the pending continuation (no next chunk launches), and this
329                        # drain point frees the chunk's pages/slots exactly once.
330                        self._free_req_resources(req)
331                    continue
```

So L set by a tracked intermediate forward lives on the ChunkedReq object and is never
committed; the final chunk (a plain `Req`, is_chunked=False) is the only req reaching line
398. This is exactly why the L drop at try_add_one loses the boundary.

Finish entry: `scheduler.py:627-639` `_free_req_resources` ->
`scheduler.py:637` `self.cache_manager.cache_req(req, finished=True)` (reached for EOS/length
finish at 385-387 and for aborted ChunkedReq at 330 - both donate paths read L).

## 8. Producer: linear.py boundary track

Anchor ~116-131. Actual: producer fn `_build_track_metadata` (def at
`python/freetoken/attention/linear.py:94`, called at 82); the c<1 `continue` is 119-120,
`boundary` computed at 122, L written at 127. Drift: -1..-2 vs anchor numbers. CHUNK_SIZE is
64 (`python/freetoken/kernel/fla/chunk.py:28`: `CHUNK_SIZE = 64`), so the anchor formula
`cached_len + (extend_len-1)//64*64` matches `r.cached_len + c * CHUNK_SIZE`.

```
113    for i, r in enumerate(reqs):
114        if r.mamba_ping_pong is None:
115            continue
116        # deepest mid-chunk boundary strictly inside the extend (h has the per-chunk state;
117        # the exact extend-end / aligned-final state lives in the live slot -> finish-donate).
118        c = (r.extend_len - 1) // CHUNK_SIZE
119        if c < 1:
120            continue
121        off = int(cu_host[i])
122        boundary = r.cached_len + c * CHUNK_SIZE
123        dst.append(r.mamba_ping_pong[r.mamba_next_track_idx])
124        h_row.append(boh[i] + c)
125        conv_src.append([off + c * CHUNK_SIZE - km1 + j for j in range(km1)])
126        boundary_rows.append(off + c * CHUNK_SIZE)
127        r.mamba_last_track_seqlen = boundary
128        r.mamba_next_track_idx = 1 - r.mamba_next_track_idx
```

Producer semantics relevant to the fix: L is ASSIGNED (overwritten), never accumulated; a
forward with c<1 leaves a carried L untouched; each track writes
`mamba_ping_pong[mamba_next_track_idx]` then flips it, so with `next_track_idx` forwarded
alongside L the frozen-slot pairing (`1 - next_track_idx`) stays valid for the carried
boundary.

## 9. Match climb - kvcache/hybrid_radix_cache.py `match_prefix`

Anchor 76-88. Actual 76-88. Drift: NONE.

```
76    def match_prefix(self, input_ids: torch.Tensor) -> HybridMatch:
77        """Match the token prefix, then truncate the reusable length to the deepest node on
78        the path that still owns a LIVE snapshot (a continuation can only resume the GDN
79        recurrence from a checkpointed boundary)."""
80        node, _ = self._walk(input_ids)
81        # walk up to the deepest node whose END boundary has a live snapshot
82        cur, end_len = node, self._path_len(node)
83        while not cur.is_root():
84            if cur.mamba_value is not None:
85                return HybridMatch(self._collect_kv(cur), end_len, cur.mamba_value, cur)
86            end_len -= cur.length
87            cur = cur.parent
88        return HybridMatch(self.empty, 0, None, self.root)
```

No live GDN snapshot anywhere on the path -> cached_len=0 even with intact KV pages (the
measured MISS). With the fix the repeat is expected to match L (e.g. 65472 for the
65585@4096 case), not the full 65536/65585.

## 10. Exhaustive consumer/producer list + early-L safety verdict

`search_for_text "mamba_last_track_seqlen"` over python/freetoken returns 10 hits in 4 files;
plus the two cache_req entry points that route into them. Verdicts for "continuation Req
carries L earlier":

| # | Site | Role | Verdict |
|---|------|------|---------|
| 1 | attention/linear.py:127 (`_build_track_metadata`) | producer: L = boundary of tracked forward | early-L SAFE - overwrites carried L when it tracks; when c<1 (119-120) it intentionally leaves L untouched, which is the fix's mechanism |
| 2 | core.py:52 | field default None | inert (the value the fix populates) |
| 3 | scheduler/cache.py:154 | docstring: L reused as pending-donate mark by the anchor | inert; documents the semantic coupling to preserve |
| 4 | scheduler/cache.py:165,172 (`snapshot_toolcall_anchor`) | consume-if-None guard; sets L=anchor | early-L SAFE with one caveat - requires L==None by decode entry. Guaranteed: every commit path that consumes carried L clears it (390 misaligned-skip, 414 donate) before the req reaches decode; the finish path drops the req. Never let a change leak a set L into decode, or tool-call anchors are silently skipped |
| 5 | scheduler/cache.py:330 | docstring of _cache_req_hybrid | inert |
| 6 | scheduler/cache.py:348-354 (finish-frozen donate, finished=True) | consumer | early-L SAFE - the intended beneficiary. `0 < L <= req.cached_len` always holds for a carried L (cached_len only grows); frozen slot `1 - next_track_idx` is the slot that wrote L because next_track_idx is forwarded with it. L not cleared here, but the req is being freed (scheduler.py:387/637/330) so no re-donate |
| 7 | scheduler/cache.py:383-414 (chunk-commit donate, finished=False) | consumer | early-L SAFE - with the fix this path becomes reachable with a carried L at the final-plain-Req commit and donates at it; L cleared at 390/414 so no double-donation and no leak into decode. Alignment skip (386-391) keeps its clear-on-skip semantics |
| 8 | scheduler/scheduler.py:322-331 | intermediate-chunk skip gate | inert w.r.t. L, but the reason forwarding is required; the fix must NOT touch it |
| 9 | scheduler/scheduler.py:398 | commit entry finished=False | safe - routes reqs 6/7 above |
| 10 | scheduler/scheduler.py:637 | commit entry finished=True via _free_req_resources (finish + abort) | safe - aborted mid-chunk ChunkedReq now also donates its carried/forwarded L; slots drained, ping-pong valid, so the donate is correct; freed exactly once (table_idx==-1 guard, scheduler.py:628-630) |

No consumer assumes "L is None until finish" in a way that breaks, and no site has
clear-on-read semantics that would consume the carried L prematurely: the only
read-and-clear sites are the two donate paths, which run exactly once per req (commit at 398
for the final Req, or free at 637). `mamba_restore_src` (prefill.py:243, consumed once at
scheduler.py:623-625) is a different field and is deliberately not forwarded - no interaction.
Non-hybrid managers (DSV4/naive, `_cache_req_swa` at cache.py:416-488) never read L (no hits).

## 11. Other continuation-Req construction sites

NONE. Evidence:
- `search_for_text "ChunkedReq"` over python/freetoken: only utils.py:11,19 (import/type),
  prefill.py:31 (class def), 36 (NotImplementedError), 197 (CLS pick), 317 (isinstance),
  scheduler.py:36 (import), 322 (isinstance).
- `search_for_text "= Req("` over python/freetoken/scheduler: zero hits - no direct Req
  construction anywhere in the scheduler package.
- Decode handoff reuses the same Req object (no reconstruction); rollback/rebuild paths
  (scheduler.py:642-752, cache.py `rebuild` 551-567) rebuild pools/page tables, not Req
  objects.

So the forwarding fix has exactly one site (below) and no other path needs the same change.

## Fix surface (exact)

`python/freetoken/scheduler/prefill.py`, `PrefillAdder.try_add_one` continuation branch:
- line 243/244 area: add `mamba_last_track_seqlen=chunked_req.mamba_last_track_seqlen,`
  to the `_add_one_req(...)` call (234-245);
- `_add_one_req` signature (140-144): add param `mamba_last_track_seqlen: int | None = None`;
- `_add_one_req` body: assign on the Req next to line 227 (e.g.
  `req.mamba_last_track_seqlen = mamba_last_track_seqlen`).
Fresh-admit branch (247-268) keeps the default None. ~4-6 lines. Do NOT change donation
semantics, ping-pong ownership, page-size or alignment conditions (comments at cache.py
340-346 and 363-367 are correct by design).

## Risks / open questions

1. Donation length at repeat: with the fix the repeat matches L (last tracked boundary, e.g.
   65472 for 65585@4096), NOT 65536/65585 - #cached-token ~65472 is the pass signal, full 65k
   equality is not expected. TASK.md hardware check already says "~65472-65536".
2. Tool-call anchor coupling (cache.py:165): safe today only because both commit paths clear
   L. Any future change that skips the final prefill commit would leak L into decode and
   silently disable anchors. Worth a one-line test asserting L is None on a req entering decode.
3. Aborted mid-chunk ChunkedReq (scheduler.py:326-330) now also finish-donates its carried L
   (correct: frozen slot + KV pages are valid and the donation is the same code path as an
   EOS finish). Behavior change is intentional but should be covered by the kvcache-level test.
4. Overlap mode: the intermediate-skip comment (scheduler.py:323-325) cites overlap
   double-cache safety; the fix does not touch commit flow, only field forwarding, so no
   interaction. Still, run the unit tests plus one hardware boot with overlap defaults.
5. `swa_evicted_seqlen` forwarding shows the established pattern to copy; `restore_src=None`
   for continuations must stay as is (live state already restored).

## Anchor drift summary (vs TASK.md numbers)

| Anchor | Actual | Drift |
|--------|--------|-------|
| cache.py finish-frozen donate ~346-362 | 347-362 (L at 348, guard 349-354, insert 357-358) | +2 |
| cache.py design comment ~343-350 | 340-346 (plus live-donate alignment comment 363-367) | -3 |
| cache.py tool-call anchor ~148-173 | 148-173 | none |
| cache.py ping-pong alloc + eviction race ~398-401 | race comment 398-400, re-lock 401, alloc 409-413 | none |
| prefill.py try_add_one ~236-250 | try_add_one 230-268, continuation branch 234-245, fields 240-244 | -2 |
| core.py Req default ~52 | core.py:52 (python/freetoken/core.py) | none |
| scheduler.py:398 final-batch commit | scheduler/scheduler.py:398 exact | none |
| linear.py ~116-131 | c/continue 118-120, boundary 122, L set 127 (fn `_build_track_metadata`, def 94) | -1..-2 |
| hybrid_radix_cache.py 76-88 | 76-88 | none |
