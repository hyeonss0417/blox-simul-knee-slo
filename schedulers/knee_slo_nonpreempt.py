"""
Knee-SLO Non-Preemptive variant.

Motivation: Inference workloads are notoriously hard to preempt — preempting
a running diffusion/LLM step usually means discarding partial work and the
client sees only the latency. The handoff doc explicitly flagged this as
a problem in v1 (Blox simulator freely suspends jobs every round, but
"real" inference rarely does).

This scheduler preserves the Knee-SLO scoring logic but ensures the
"already-running" jobs (attained_service > 0) keep their slot at the top
of the order until they finish. Newly arrived / suspended jobs compete
for the *remaining* slots using the standard Knee score.
"""
from .knee_slo import KneeSlo
from .scheduler_policy import SchedulingPolicy
import pandas as pd
from typing import Optional


class KneeSloNonPreempt(KneeSlo):

    @SchedulingPolicy.copy_arguments
    def schedule(
        self,
        job_dict: dict,
        node_info: dict,
        gpu_df: pd.DataFrame,
        global_placement_policy: Optional[str] = None,
    ) -> dict:
        # First compute the standard Knee score for tie-breaking.
        parent = self._schedule_inner(job_dict, node_info, gpu_df)
        scored = parent["job_order"]

        # Partition: running jobs first (in their existing order, by
        # remaining work ascending to drain fast), then queued jobs by
        # Knee score.
        running = []
        queued = []
        for jid, job in scored:
            attained = job.get("tracked_metrics", {}).get("attained_service", 0)
            if attained > 0:
                running.append((jid, job))
            else:
                queued.append((jid, job))

        # Among running jobs: keep low slo_risk first (urgency-aware).
        running.sort(key=lambda x: x[1].get("slo_risk", 0.0), reverse=True)
        # Queued already sorted by Knee score from super().

        return {
            "job_order": running + queued,
            "run_all_jobs": False,
        }
