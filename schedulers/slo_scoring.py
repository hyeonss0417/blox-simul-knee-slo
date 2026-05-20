"""
SLO-Aware Asymmetric Scoring Scheduler

Implements the scheduling policy from the research proposal:
  - Jobs approaching SLO deadline get exponentially higher priority (penalty)
  - Jobs with slack get linearly lower priority (reward)
  - This asymmetry prevents SLO violations while maintaining fairness

Scoring function (from proposal Eq. 2):
  score(slack) = -alpha * |slack|   if slack < 0  (SLO violation imminent)
               =  beta * slack      if slack >= 0 (has time to spare)

where:
  slack = SLO_deadline - (current_time + estimated_remaining_time)
  alpha >> beta  (asymmetry: penalize violations much more than reward slack)
"""
from .scheduler_policy import SchedulingPolicy
import pandas as pd
from typing import Optional


class SloScoring(SchedulingPolicy):
    """
    SLO-aware scheduler using asymmetric scoring.
    Jobs closer to or past their SLO deadline are prioritized.
    """

    def __init__(self, args):
        self.metric_to_track = ["per_iter_time", "attained_service"]
        self.default_metric_value = [0, 0]
        # Asymmetry parameters
        self.alpha = 10.0  # penalty weight for SLO violations
        self.beta = 1.0    # reward weight for slack
        # Default SLO multiplier: deadline = submit_time + slo_multiplier * estimated_duration
        self.slo_multiplier = 2.0

    def _compute_slo_deadline(self, job):
        """Compute SLO deadline for a job."""
        estimated_duration = job["job_iteration_time"] * job["job_total_iteration"]
        submit_time = job.get("submit_time", 0)
        return submit_time + self.slo_multiplier * estimated_duration

    def _compute_remaining_time(self, job):
        """Estimate remaining execution time."""
        total_work = job["job_iteration_time"] * job["job_total_iteration"]
        attained = job.get("tracked_metrics", {}).get("attained_service", 0)
        return max(0, total_work - attained)

    def _asymmetric_score(self, slack):
        """
        Asymmetric scoring function.
        Lower score = higher priority.
        Negative slack (violation) -> large negative score (highest priority).
        Positive slack (safe) -> small positive score (lower priority).
        """
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
        """
        Schedule jobs based on SLO-aware asymmetric scoring.
        Jobs with lowest score (most urgent) are scheduled first.
        """
        for job_id in job_dict:
            job = job_dict[job_id]
            deadline = self._compute_slo_deadline(job)
            remaining = self._compute_remaining_time(job)
            current_time = job.get("tracked_metrics", {}).get("attained_service", 0) + job.get("submit_time", 0)
            slack = deadline - (current_time + remaining)
            job["slo_score"] = self._asymmetric_score(slack)

        # Sort by priority first, then by SLO score (lower = more urgent)
        sorted_job_order = sorted(
            job_dict.items(),
            key=lambda x: (x[1]["job_priority"], x[1]["slo_score"]),
        )

        schedule_info = dict()
        schedule_info["job_order"] = sorted_job_order
        schedule_info["run_all_jobs"] = False
        return schedule_info
