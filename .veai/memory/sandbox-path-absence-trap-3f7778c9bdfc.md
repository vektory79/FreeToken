---
name: "sandbox-path-absence-trap"
description: "Delegated agents' list_dir is sandboxed to project root; outside paths falsely reported absent - verify via IDE terminal"
type: project
lastUpdated: 2026-09-23T20:10
lastRecall: 2026-09-26T18:28
---

# Sandbox tool false-absence trap: ask-agent list_dir sees only project root

2026-09-23: an ask agent reported "/media/ai/models does not exist on this machine" - FALSE. The built-in list_dir/file tools in delegated agents are sandboxed to the project root (/media/ai/src/FreeToken) and report paths outside it as nonexistent, even /media/ai/src. The project sits at /media/ai/src/FreeToken while model checkpoints live at /media/ai/models/ (RadixArk, unsloth, RedHatAI) - outside the sandbox root.

**Why:** the agent trusted its own tool's absence signal instead of cross-checking against known-good evidence (hardware arms had loaded GLM checkpoints from /media/ai/models days earlier).

**How to apply:** never conclude a path outside the project root is absent based on sandboxed list_dir/read tools; verify through an escape hatch - IDE terminal MCP (run_command/ls), or ask the user. Treat "does not exist" claims about /media/ai/models, /media/ai/src sibling dirs, or any absolute path outside the repo as unverified until cross-checked.
