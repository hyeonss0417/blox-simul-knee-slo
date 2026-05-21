"""
LAS-SLO: LAS base + SLO emergency bucket.

Key idea — beat LAS on SLO miss rate WITHOUT damaging Avg JCT.

LAS itself ignores deadlines and just sorts by attained_service.  It wins
on Avg JCT but has no protection for jobs about to miss SLO.

LAS-SLO adds a small "critical lane":

    bucket =
        0  if  wait_time >= slo_target               # already missed
        1  if  wait_time >= θ × slo_target           # close to miss
        2  otherwise                                  # safe — pure LAS

Within bucket 0/1: most-overdue first (urgency).
Within bucket 2: attained_service ascending (LAS).

Sort key = (job_priority, bucket, secondary_key).  Bucket comparison
takes precedence, so a single newly-overdue job jumps ahead of fresh
short jobs — but most of the time bucket==2 and behavior is identical
to LAS.

This avoids the urgency-saturation pathology of Knee because urgency is
only used for tie-breaking within bucket 0, not as a global -w_urg term
that competes with size.

Tunable via env vars:
    LAS_SLO_TARGET       SLO target seconds (default 86400 = 24h)
    LAS_SLO_THETA        warning-zone fraction (default 0.7)
"""
from .scheduler_policy import SchedulingPolicy
import os
import pandas as pd
from typing import Optional


class LasSlo(SchedulingPolicy):
    def __init__(self, args):
        self.metric_to_track = ["per_iter_time", "attained_service"]
        self.default_metric_value = [0, 0]
        self.current_time = 0
        self.slo_target = float(os.environ.get("LAS_SLO_TARGET", "86400"))
        self.theta = float(os.environ.get("LAS_SLO_THETA", "0.7"))

    @SchedulingPolicy.copy_arguments
    def schedule(
        self,
        job_dict: dict,
        node_info: dict,
        gpu_df: pd.DataFrame,
        global_placement_policy: Optional[str] = None,
    ) -> dict:
        now = float(self.current_time)
        warning_threshold = self.theta * self.slo_target

        for jid, job in job_dict.items():
            attained = job.get("tracked_metrics", {}).get("attained_service", 0)
            submit = float(job.get("submit_time", 0))
            wait = now - submit

            if wait >= self.slo_target:
                bucket = 0
                # most overdue first → larger wait = smaller (negative) score
                secondary = -wait
            elif wait >= warning_threshold:
                bucket = 1
                secondary = attained
            else:
                bucket = 2
                secondary = attained  # pure LAS

            job["las_slo_score"] = (bucket, secondary)

        sorted_order = sorted(
            job_dict.items(),
            key=lambda x: (x[1]["job_priority"], x[1]["las_slo_score"]),
        )
        return {"job_order": sorted_order, "run_all_jobs": False}
