"""
SJF-total-work baseline.

Sorts by predicted *total* work (category average), not remaining.
This is an important baseline because v1 SloScoring was implicitly
behaving like this — the v2 report explicitly contrasts SJF-total-work
with Knee-SLO to show the SLO-aware uplift.
"""
from .scheduler_policy import SchedulingPolicy
import pandas as pd
from collections import defaultdict
from typing import Optional


class SjfTotal(SchedulingPolicy):
    def __init__(self, args):
        self.metric_to_track = ["per_iter_time", "attained_service"]
        self.default_metric_value = [0, 0]
        self.current_time = 0
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

    def _predicted_total(self, job):
        cat = job.get("job_class_id", -1)
        if cat in self._cat_avg:
            return self._cat_avg[cat]
        if self._global_avg is not None:
            return self._global_avg
        return job.get("job_total_iteration", 300) * job.get("job_iteration_time", 1)

    @SchedulingPolicy.copy_arguments
    def schedule(
        self,
        job_dict: dict,
        node_info: dict,
        gpu_df: pd.DataFrame,
        global_placement_policy: Optional[str] = None,
    ) -> dict:
        for jid, job in job_dict.items():
            job["pred_total"] = self._predicted_total(job)

        sorted_job_order = sorted(
            job_dict.items(),
            key=lambda x: (x[1]["job_priority"], x[1]["pred_total"]),
        )
        return {"job_order": sorted_job_order, "run_all_jobs": False}
