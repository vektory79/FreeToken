#!/usr/bin/env python3
r"""Markdown table linter for a knowledge base (kb/).

Finds the two table-breakage classes observed in kb/ (2026-09):
1. a table broken by blank lines between rows (each row renders as a fragment);
2. column-count mismatches (header vs separator vs rows), respecting escaped
   pipes `\|` inside cell text.

Usage:
    python3 table_lint.py [path-to-kb-dir]

Default path: ./kb relative to the current working directory (run from the
repo root). Exit code: 0 = no issues, 1 = issues found (usable as a gate in
the kb-distill reflex loop).
"""
import pathlib
import re
import sys

PIPE = re.compile(r"^\s*\|.*\|\s*$")
SEP = re.compile(r"^\s*\|(\s*:?-{2,}:?\s*\|)+\s*$")


def cells(line):
    s = line.strip().strip("|")
    return [c.strip() for c in re.split(r"(?<!\\)\|", s)]


def is_sep(line):
    return bool(SEP.match(line)) and all(re.fullmatch(r":?-{2,}:?", c) for c in cells(line))


def main():
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "kb")
    if not root.is_dir():
        print(f"ERROR: kb dir not found: {root}")
        return 1
    issues = []
    for md in sorted(root.rglob("*.md")):
        lines = md.read_text(encoding="utf-8", errors="replace").splitlines()
        pipe_idx = [i for i, l in enumerate(lines) if PIPE.match(l)]
        if not pipe_idx:
            continue
        blocks, cur = [], [pipe_idx[0]]
        for i in pipe_idx[1:]:
            if i == cur[-1] + 1:
                cur.append(i)
            else:
                blocks.append(cur)
                cur = [i]
        blocks.append(cur)
        for block in blocks:
            first = block[0]
            if is_sep(lines[first]):
                issues.append((md, first + 1, "table starts with separator, no header row"))
            ref = None
            if len(block) >= 2 and is_sep(lines[block[1]]):
                ref = cells(lines[first])
                sep_cells = cells(lines[block[1]])
                if len(ref) != len(sep_cells):
                    issues.append((md, block[1] + 1,
                                   f"separator has {len(sep_cells)} cols, header has {len(ref)}"))
            for ln in block:
                if is_sep(lines[ln]):
                    continue
                c = cells(lines[ln])
                if ref is not None and len(c) != len(ref):
                    issues.append((md, ln + 1,
                                   f"row has {len(c)} cols, header has {len(ref)}: {lines[ln][:80]}"))
        for a, b in zip(blocks, blocks[1:]):
            if b[0] - a[-1] == 2 and not lines[a[-1] + 1].strip():
                issues.append((md, a[-1] + 2,
                               f"blank line breaks table: block ending at L{a[-1]+1} and block starting at L{b[0]+1}"))

    if not issues:
        print(f"OK: no table issues in {root}")
        return 0
    for md, ln, msg in issues:
        print(f"{md.relative_to(root)}:{ln}: {msg}")
    print(f"\ntotal issues: {len(issues)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())