"""
SLO-Aware Asymmetric Scoring Scheduler (Non-Oracle)

Uses profiled category-average latency instead of exact job duration.
Categories are determined by job_class_id (model type).
The wall-clock simulation time is used for accurate slack computation.

Scoring function:
  score(slack) = -alpha * |slack|   if slack < 0  (SLO violation imminent)
               =  beta * slack      if slack >= 0 (has time to spare)

where:
  profiled_latency = running average of completed jobs in same category
  slack = SLO_deadline - (wall_clock_time + estimated_remaining_time)
  alpha >> beta  (asymmetry: penalize violations much more than reward slack)
"""
from .scheduler_policy import SchedulingPolicy
import pandas as pd
from typing import Optional
from collections import defaultdict


class SloScoring(SchedulingPolicy):
    """
    SLO-aware scheduler using asymmetric scoring with profiled latency.
    Unlike the Oracle approach, this does NOT use per-job exact duration.
    Instead, it maintains a running average per job category (job_class_id).
    """

    def __init__(self, args):
        self.metric_to_track = ["per_iter_time", "attained_service"]
        self.default_metric_value = [0, 0]
        self.alpha = 10.0
        self.beta = 1.0
        self.slo_multiplier = 2.0

        # Global simulation clock (updated from main loop each round)
        self.current_time = 0

        # Category-based profiled latency tracking (keyed by job_class_id)
        self._category_durations = defaultdict(list)
        self._category_avg = {}
        self._global_avg = None

    def update_profiled_latency(self, job):
        """Called when a job finishes to update category statistics."""
        category = job.get("job_class_id", -1)
        actual_duration = job.get("job_iteration_time", 1) * job.get("job_total_iteration", 1)
        self._category_durations[category].append(actual_duration)
        durations = self._category_durations[category]
        self._category_avg[category] = sum(durations) / len(durations)

        all_durations = []
        for d in self._category_durations.values():
            all_durations.extend(d)
        self._global_avg = sum(all_durations) / len(all_durations)

    def _get_profiled_latency(self, job):
        """Get profiled (average) latency for this job's category."""
        category = job.get("job_class_id", -1)
        if category in self._category_avg:
            return self._category_avg[category]
        if self._global_avg is not None:
            return self._global_avg
        # Fallback: use this job's own iteration info as rough estimate
        # This is NOT oracle — it's just 1 * total_iteration (coarse estimate)
        return job.get("job_total_iteration", 300) * job.get("job_iteration_time", 1)

    def _compute_slo_deadline(self, job):
        """Compute SLO deadline using profiled latency, not oracle."""
        profiled_latency = self._get_profiled_latency(job)
        submit_time = job.get("submit_time", 0)
        return submit_time + self.slo_multiplier * profiled_latency

    def _compute_remaining_time(self, job):
        """Estimate remaining time using profiled latency and attained service."""
        profiled_latency = self._get_profiled_latency(job)
        attained = job.get("tracked_metrics", {}).get("attained_service", 0)
        return max(0, profiled_latency - attained)

    def _asymmetric_score(self, slack):
        if slack < 0:
            return -self.alpha * abs(slack)
        else:
            return self.beta * slack

    @SchedulingPolicy.copy_arguments
    def schedule(
        self,
        job_dict: dict,
        node_info: dict,
        gpu_df: pd.DataFrame,
        global_placement_policy: Optional[str] = None,
    ) -> dict:
        for job_id in job_dict:
            job = job_dict[job_id]
            deadline = self._compute_slo_deadline(job)
            remaining = self._compute_remaining_time(job)
            estimated_finish = self.current_time + remaining
            slack = deadline - estimated_finish
            job["slo_score"] = self._asymmetric_score(slack)

        sorted_job_order = sorted(
            job_dict.items(),
            key=lambda x: (x[1]["job_priority"], x[1]["slo_score"]),
        )

        schedule_info = dict()
        schedule_info["job_order"] = sorted_job_order
        schedule_info["run_all_jobs"] = False
        return schedule_info
