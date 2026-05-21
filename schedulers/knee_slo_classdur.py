"""
Knee-SLO with per-class SLO target scaled by category-average duration.

KneeSloClass uses fixed-tier targets (30min / 2h / 6h) based on
`class_id mod 3` — synthetic but artificial.  KneeSloClassDur instead
uses the *measured* category-mean duration (the same statistic the
scheduler already tracks for non-Oracle prediction):

    B_class = slo_multiplier * category_mean_duration

So large-job classes get proportionally larger budgets and small-job
classes get tighter ones — exactly the "real-world SLO tiering" the
prompt asks for, but derived from observed data rather than guessed
class labels.

Cold-start: until we have any completed job in a class, fall back to
the global slo_target_seconds default.
"""
from .knee_slo import KneeSlo
from .scheduler_policy import SchedulingPolicy
import os
import pandas as pd
from typing import Optional


class KneeSloClassDur(KneeSlo):
    DEFAULTS = dict(KneeSlo.DEFAULTS)
    DEFAULTS["slo_multiplier"] = 2.5  # B_class = 2.5 * class_mean_dur

    def __init__(self, args):
        super().__init__(args)
        # Override w/ env if set.
        if "KNEE_SLO_MULTIPLIER" in os.environ:
            try:
                self.slo_multiplier = float(os.environ["KNEE_SLO_MULTIPLIER"])
            except ValueError:
                self.slo_multiplier = 2.5
        if not hasattr(self, "slo_multiplier"):
            self.slo_multiplier = 2.5

    def _class_target(self, job):
        cat = job.get("job_class_id", -1)
        if cat in self._cat_avg:
            return self.slo_multiplier * self._cat_avg[cat]
        if self._global_avg is not None:
            return self.slo_multiplier * self._global_avg
        return self.slo_target_seconds

    @SchedulingPolicy.copy_arguments
    def schedule(
        self,
        job_dict: dict,
        node_info: dict,
        gpu_df: pd.DataFrame,
        global_placement_policy: Optional[str] = None,
    ) -> dict:
        now = float(self.current_time)
        global_avg = self._global_avg if self._global_avg else 300.0

        for jid, job in job_dict.items():
            B = float(self._class_target(job))
            submit = float(job.get("submit_time", 0))
            rem = self._predicted_remaining(job)
            pred_total = self._predicted_total(job)
            completion_risk = (now - submit + rem) / max(1.0, B)
            queue_budget = max(B - pred_total, 60.0)
            wait_risk = (now - submit) / queue_budget
            risk = max(completion_risk, wait_risk)
            urg = self._urgency(risk)
            norm_remaining = rem / max(1.0, global_avg)
            age_bonus = max(0.0, (now - submit) - self.age_threshold) / max(
                1.0, self.age_threshold
            )
            score = (
                self.w_size * norm_remaining
                - self.w_urg * urg
                - self.w_age * age_bonus
            )
            job["slo_score"] = score
            job["slo_risk"] = risk
            job["slo_budget"] = B

        sorted_order = sorted(
            job_dict.items(),
            key=lambda x: (x[1]["job_priority"], x[1]["slo_score"]),
        )
        return {"job_order": sorted_order, "run_all_jobs": False}
