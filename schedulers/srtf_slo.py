"""
SRTF-SLO: SRTF base (oracle remaining) + SLO emergency bucket.

Same bucket trick as LasSlo but uses oracle predicted_remaining as the
safe-zone secondary key.  Should match SRTF on Avg JCT and beat it on
SLO miss rate.

Note: predicted_remaining here uses category-mean (non-Oracle), so it's
really "SJF-total-with-SLO-lane" — but the structure is identical.
"""
from .scheduler_policy import SchedulingPolicy
import os
import pandas as pd
from collections import defaultdict
from typing import Optional


class SrtfSlo(SchedulingPolicy):
    def __init__(self, args):
        self.metric_to_track = ["per_iter_time", "attained_service"]
        self.default_metric_value = [0, 0]
        self.current_time = 0
        self.slo_target = float(os.environ.get("SRTF_SLO_TARGET", "86400"))
        self.theta = float(os.environ.get("SRTF_SLO_THETA", "0.7"))
        self._cat_durations = defaultdict(list)
        self._cat_avg = {}
        self._global_avg = None

    def update_profiled_latency(self, job):
        c = job.get("job_class_id", -1)
        d = job.get("job_iteration_time", 1) * job.get("job_total_iteration", 1)
        self._cat_durations[c].append(d)
        self._cat_avg[c] = sum(self._cat_durations[c]) / len(self._cat_durations[c])
        all_d = []
        for v in self._cat_durations.values():
            all_d.extend(v)
        self._global_avg = sum(all_d) / max(1, len(all_d))

    def _predicted_remaining(self, job):
        c = job.get("job_class_id", -1)
        total = self._cat_avg.get(c, self._global_avg or
                                  job.get("job_total_iteration", 300) *
                                  job.get("job_iteration_time", 1))
        attained = job.get("tracked_metrics", {}).get("attained_service", 0)
        return max(0.0, total - attained)

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
            submit = float(job.get("submit_time", 0))
            wait = now - submit
            remaining = self._predicted_remaining(job)

            if wait >= self.slo_target:
                bucket = 0
                secondary = -wait  # most overdue first
            elif wait >= warning_threshold:
                bucket = 1
                secondary = remaining  # SRTF within warning
            else:
                bucket = 2
                secondary = remaining  # SRTF within safe

            job["srtf_slo_score"] = (bucket, secondary)

        sorted_order = sorted(
            job_dict.items(),
            key=lambda x: (x[1]["job_priority"], x[1]["srtf_slo_score"]),
        )
        return {"job_order": sorted_order, "run_all_jobs": False}
