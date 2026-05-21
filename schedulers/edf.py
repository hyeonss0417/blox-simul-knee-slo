"""
EDF: Earliest Deadline First.

Uses absolute SLO target (default 6h; configurable via env EDF_SLO_TARGET).
"""
from .scheduler_policy import SchedulingPolicy
import os
import pandas as pd
from typing import Optional


class Edf(SchedulingPolicy):
    def __init__(self, args):
        self.metric_to_track = ["per_iter_time", "attained_service"]
        self.default_metric_value = [0, 0]
        self.current_time = 0
        self.slo_target_seconds = float(
            os.environ.get("EDF_SLO_TARGET", 21600.0)
        )  # 6h default

    @SchedulingPolicy.copy_arguments
    def schedule(
        self,
        job_dict: dict,
        node_info: dict,
        gpu_df: pd.DataFrame,
        global_placement_policy: Optional[str] = None,
    ) -> dict:
        T = self.slo_target_seconds
        for jid, job in job_dict.items():
            submit = float(job.get("submit_time", 0))
            job["deadline"] = submit + T

        sorted_job_order = sorted(
            job_dict.items(),
            key=lambda x: (x[1]["job_priority"], x[1]["deadline"]),
        )
        return {"job_order": sorted_job_order, "run_all_jobs": False}
