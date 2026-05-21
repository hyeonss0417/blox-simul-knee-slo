"""
HRRN (Highest Response Ratio Next).

Classical OS scheduling policy:

    R = (wait + service) / service

Job with the highest R wins.  Combines SJF's bias toward short jobs with
LAS-like aging — a long job's R grows over time as wait increases, so
it eventually beats short newcomers.  Useful baseline against KneeSlo:
HRRN gets aging+size for free without an SLO budget.
"""
from .scheduler_policy import SchedulingPolicy
import pandas as pd
from collections import defaultdict
from typing import Optional


class Hrrn(SchedulingPolicy):
    def __init__(self, args):
        self.metric_to_track = ["per_iter_time", "attained_service"]
        self.default_metric_value = [0, 0]
        self.current_time = 0
        self._cat_durations = defaultdict(list)
        self._cat_avg = {}
        self._global_avg = None

    def update_profiled_latency(self, job):
        cat = job.get("job_class_id", -1)
        d = job.get("job_iteration_time", 1) * job.get("job_total_iteration", 1)
        self._cat_durations[cat].append(d)
        self._cat_avg[cat] = sum(self._cat_durations[cat]) / len(self._cat_durations[cat])
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
        now = float(self.current_time)
        for jid, job in job_dict.items():
            submit = float(job.get("submit_time", 0))
            wait = max(0.0, now - submit)
            service = max(1.0, self._predicted_total(job))
            R = (wait + service) / service
            job["hrrn_score"] = -R  # higher R = lower (better) sort key

        sorted_order = sorted(
            job_dict.items(),
            key=lambda x: (x[1]["job_priority"], x[1]["hrrn_score"]),
        )
        return {"job_order": sorted_order, "run_all_jobs": False}
