from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Set

from freetoken.core import Batch, Req
from freetoken.utils import init_logger

logger = init_logger(__name__)


@dataclass
class DecodeManager:
    page_size: int
    # slots a decode step may hold past each request's committed length (provisional rows)
    token_headroom: int = 0
    running_reqs: Set[Req] = field(default_factory=set)
    # per-request route affinity: the ladder of captured decode graphs and the server
    # default. An empty ladder means no per-request snapping (the server default everywhere).
    route_affinity_ladder: tuple[float, ...] = ()
    route_affinity_default: float = 0.0

    def filter_reqs(self, reqs: Iterable[Req]) -> None:
        self.running_reqs = {req for req in self.running_reqs.union(reqs) if req.can_decode}

    def remove_req(self, req: Req) -> None:
        self.running_reqs.discard(req)

    def abort_req(self, uid: int) -> Req | None:
        for req in self.running_reqs:
            if req.uid == uid:
                self.running_reqs.remove(req)
                return req
        return None

    @property
    def inflight_tokens(self) -> int:
        # each request's partly filled page, plus the provisional rows a decode step may add
        tokens_reserved = (self.page_size - 1 + self.token_headroom) * len(self.running_reqs)
        return sum(req.remain_len for req in self.running_reqs) + tokens_reserved

    def schedule_next_batch(self) -> Batch | None:
        if not self.runnable:
            return None
        batch = Batch(reqs=sorted(self.running_reqs, key=lambda req: req.uid), phase="decode")
        # local import: engine.config drags the whole engine package in; decode.py is
        # imported light (worker + test harnesses), so pay for it on first use instead
        from freetoken.engine.config import batch_route_affinity

        value, mismatch = batch_route_affinity(
            [req.route_affinity for req in batch.reqs],
            ladder=self.route_affinity_ladder,
            default=self.route_affinity_default,
        )
        batch.route_affinity = value
        if mismatch:
            logger.warning_once(
                "decode-batch requests asked for different route_affinity values; decoding at 0.0"
            )
        return batch

    @property
    def runnable(self) -> bool:
        return len(self.running_reqs) > 0
