"""
HrrnSlo — HRRN base + SLO emergency bucket.

HRRN (Highest Response Ratio Next): sort by R = (wait + service) / service ↓.
This naturally gives:
- Short jobs (small service): high R → priority
- Old jobs (large wait): high R → priority (aging)

Combined with SLO bucket:
- bucket 0 (critical): wait >= SLO_target → -wait (most overdue first)
- bucket 1 (warning): wait >= θ·SLO_target → HRRN R
- bucket 2 (safe):    otherwise → HRRN R

Predicted service uses metadata when available, else category mean.
"""
from .scheduler_policy import SchedulingPolicy
from .meta_pred import _predicted_dur
import os
import pandas as pd
from typing import Optional


class HrrnSlo(SchedulingPolicy):
    def __init__(self, args):
        self.metric_to_track = ["per_iter_time", "attained_service"]
        self.default_metric_value = [0, 0]
        self.current_time = 0
        self.slo_target = float(os.environ.get("HRRN_SLO_TARGET", "300"))
        self.theta = float(os.environ.get("HRRN_SLO_THETA", "0.7"))
        self.use_meta = os.environ.get("HRRN_USE_META", "1") == "1"

    def _service_estimate(self, job):
        if self.use_meta:
            return max(1.0, _predicted_dur(job))
        return max(1.0, job.get("job_total_iteration", 30) *
                       job.get("job_iteration_time", 1))

    @SchedulingPolicy.copy_arguments
    def schedule(
        self,
        job_dict: dict,
        node_info: dict,
        gpu_df: pd.DataFrame,
        global_placement_policy: Optional[str] = None,
    ) -> dict:
        now = float(self.current_time)
        warning = self.theta * self.slo_target

        for jid, job in job_dict.items():
            submit = float(job.get("submit_time", 0))
            wait = max(0.0, now - submit)
            service = self._service_estimate(job)
            R = (wait + service) / service

            # All buckets use -R so HrrnSlo never regresses below HRRN
            # when the SLO threshold is over-tripped (a common failure mode in
            # heavy-tail mixed workloads where most jobs land in bucket 0).
            # Bucket separation only matters when comparing across buckets.
            if wait >= self.slo_target:
                bucket = 0
                secondary = -R        # critical: still HRRN-ranked
            elif wait >= warning:
                bucket = 1
                secondary = -R
            else:
                bucket = 2
                secondary = -R

            job["hrrn_slo_score"] = (bucket, secondary)

        sorted_order = sorted(
            job_dict.items(),
            key=lambda x: (x[1]["job_priority"], x[1]["hrrn_slo_score"]),
        )
        return {"job_order": sorted_order, "run_all_jobs": False}
