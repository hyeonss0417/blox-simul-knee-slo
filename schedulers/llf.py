"""
LLF: Least Laxity First.

laxity = deadline - now - predicted_remaining

Uses absolute SLO target (default 6h) and category-average remaining
(non-Oracle).
"""
from .scheduler_policy import SchedulingPolicy
import os
import pandas as pd
from collections import defaultdict
from typing import Optional


class Llf(SchedulingPolicy):
    def __init__(self, args):
        self.metric_to_track = ["per_iter_time", "attained_service"]
        self.default_metric_value = [0, 0]
        self.current_time = 0
        self.slo_target_seconds = float(
            os.environ.get("LLF_SLO_TARGET", 21600.0)
        )
        self._cat_durations = defaultdict(list)
        self._cat_avg = {}
        self._global_avg = None

    def update_profiled_latency(self, job):
        cat = job.get("job_class_id", -1)
        actual = job.get("job_iteration_time", 1) * job.get("job_total_iteration", 1)
        self._cat_durations[cat].append(actual)
        durs = self._cat_durations[cat]
        self._cat_avg[cat] = sum(durs) / len(durs)
        all_d = []
        for v in self._cat_durations.values():
            all_d.extend(v)
        self._global_avg = sum(all_d) / max(1, len(all_d))

    def _predicted_remaining(self, job):
        cat = job.get("job_class_id", -1)
        attained = job.get("tracked_metrics", {}).get("attained_service", 0)
        if cat in self._cat_avg:
            total = self._cat_avg[cat]
        elif self._global_avg is not None:
            total = self._global_avg
        else:
            total = job.get("job_total_iteration", 300) * job.get(
                "job_iteration_time", 1
            )
        return max(0.0, total - attained)

    @SchedulingPolicy.copy_arguments
    def schedule(
        self,
        job_dict: dict,
        node_info: dict,
        gpu_df: pd.DataFrame,
        global_placement_policy: Optional[str] = None,
    ) -> dict:
        T = self.slo_target_seconds
        now = float(self.current_time)
        for jid, job in job_dict.items():
            submit = float(job.get("submit_time", 0))
            deadline = submit + T
            rem = self._predicted_remaining(job)
            job["laxity"] = deadline - now - rem

        sorted_job_order = sorted(
            job_dict.items(),
            key=lambda x: (x[1]["job_priority"], x[1]["laxity"]),
        )
        return {"job_order": sorted_job_order, "run_all_jobs": False}
