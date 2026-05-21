"""
MetaSrtfSlo — MetaSrtf (non-Oracle short-first) + SLO emergency bucket.

Motivation:
- MetaSrtf has the LOWEST Avg JCT but WORSE tail than oracle SRTF
  (in the 2-GPU contention test: Avg 63.2s but P99 503 vs SRTF's 388).
- SrtfSlo (with bucket) has tighter tail (P99 198) but uses oracle.
- This scheduler combines both: non-Oracle metadata prediction for
  safe-zone ordering, bucket protection for tail.

Algorithm:
    bucket =
        0 if wait_time >= SLO              (critical)
        1 if wait_time >= θ × SLO          (warning)
        2 otherwise                         (safe)
    secondary key:
        bucket 0: -wait                    (most overdue first)
        bucket 1: meta_predicted_remaining (SRTF-like with meta pred)
        bucket 2: meta_predicted_remaining (SRTF-like with meta pred)

This is the natural extension of MetaSrtf with SLO protection.
"""
from .scheduler_policy import SchedulingPolicy
from .meta_pred import _predicted_dur
import os
import pandas as pd
from typing import Optional


class MetaSrtfSlo(SchedulingPolicy):
    def __init__(self, args):
        self.metric_to_track = ["per_iter_time", "attained_service"]
        self.default_metric_value = [0, 0]
        self.current_time = 0
        self.slo_target = float(os.environ.get("META_SLO_TARGET", "60"))
        self.slo_multiplier = float(os.environ.get("META_SLO_MULT", "0"))
        self.theta = float(os.environ.get("META_SLO_THETA", "0.7"))

    @SchedulingPolicy.copy_arguments
    def schedule(
        self,
        job_dict: dict,
        node_info: dict,
        gpu_df: pd.DataFrame,
        global_placement_policy: Optional[str] = None,
    ) -> dict:
        now = float(self.current_time)
        for jid, job in job_dict.items():
            submit = float(job.get("submit_time", 0))
            wait = now - submit
            total = _predicted_dur(job)
            attained = job.get("tracked_metrics", {}).get("attained_service", 0)
            remaining = max(0.0, total - attained)

            # Per-job SLO budget — metadata-scaled if multiplier set.
            B = (self.slo_multiplier * total) if self.slo_multiplier > 0 \
                else self.slo_target

            warning = self.theta * B
            if wait >= B:
                bucket = 0
                secondary = -wait
            elif wait >= warning:
                bucket = 1
                secondary = remaining
            else:
                bucket = 2
                secondary = remaining  # MetaSrtf-like ordering

            job["meta_srtf_slo_score"] = (bucket, secondary)

        sorted_order = sorted(
            job_dict.items(),
            key=lambda x: (x[1]["job_priority"], x[1]["meta_srtf_slo_score"]),
        )
        return {"job_order": sorted_order, "run_all_jobs": False}
