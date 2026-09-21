import logging

# v0 A/B liveness probe only: the gguf moe prefill branch logs its one-shot
# marker via a bare stdlib logger that no server handler covers, so INFO
# records are dropped by logging.lastResort. A root handler makes them
# reachable. Harness-side; production code untouched.
logging.basicConfig(level=logging.INFO)
