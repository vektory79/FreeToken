#!/usr/bin/env python3
"""Probe: mul_mat_q8_0 launch sequence inside one steady chunk.

Splits the capture into chunks by >1 s gaps in kernel start times, takes a
middle full chunk, and prints every mul_mat_q8_0 launch (grid, ms) in order so
each launch maps to its op class by position within the per-layer pattern.
"""
import os
import sqlite3

TASKDIR = os.path.dirname(os.path.abspath(__file__))
con = sqlite3.connect(os.path.join(TASKDIR, "denseq80.sqlite"))
cur = con.cursor()

rows = cur.execute("""
    SELECT k.start, k.end, k.gridX, k.gridY
    FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName = s.id
    WHERE s.value = 'mul_mat_q8_0'
    ORDER BY k.start
""").fetchall()

# split into chunks on >1 s gaps
chunks = []
cur_chunk = [rows[0]]
for r in rows[1:]:
    if r[0] - cur_chunk[-1][0] > 1e9:
        chunks.append(cur_chunk)
        cur_chunk = []
    cur_chunk.append(r)
chunks.append(cur_chunk)
print("n chunk-spans:", len(chunks), "| sizes:", [len(c) for c in chunks])

# pick a span whose duration ~= T (14.87 s): a full chunk
best = min(chunks, key=lambda c: abs((c[-1][1] - c[0][0]) / 1e9 - 14.87))
span = (best[-1][1] - best[0][0]) / 1e9
print("probe span: %d launches, %.3f s" % (len(best), span))

for i, (st, en, gx, gy) in enumerate(best):
    print("%4d  grid=(%d,%d)  %7.3f ms" % (i, gx, gy, (en - st) / 1e6))
