"""
Knee-SLO with class-aware SLO targets.

Motivation: A single global SLO target (6h) is unrealistic because
different inference workloads have wildly different latency
expectations. Image classification expects <1s; LLM generation accepts
~10s; offline batch generation tolerates hours.

We map the trace's `job_class_id` to one of three SLO tiers:

    interactive (class id ≡ 0 mod 3):  30 min   (1800 s)
    standard    (class id ≡ 1 mod 3):  2 hours  (7200 s)
    batch       (class id ≡ 2 mod 3):  6 hours  (21600 s)

The modulo split is synthetic since the trace does not provide a true
"latency tier" label — it's a stress test for class-aware scheduling.
The Knee scoring logic then operates per-class with the right denominator.
"""
from .knee_slo import KneeSlo
from .scheduler_policy import SchedulingPolicy
import pandas as pd
from typing import Optional


class KneeSloClass(KneeSlo):
    CLASS_TARGETS = {
        0: 1800.0,   # 30min — interactive
        1: 7200.0,   # 2h   — standard
        2: 21600.0,  # 6h   — batch
    }

    def _class_target(self, job):
        cat = int(job.get("job_class_id", 0))
        return self.CLASS_TARGETS[cat % 3]

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
