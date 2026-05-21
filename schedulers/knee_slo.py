"""
Knee-SLO: Normalized Deadline-Risk Scheduling for GPU Inference Workloads.

Key differences from v1 SloScoring:
1) Wall-clock simulation time (`current_time`) is fed from main loop, not derived
   from attained_service.
2) Absolute SLO target (`slo_target_seconds`) is used instead of
   `slo_multiplier × predicted_duration`, so SLO budget itself is NOT
   biased toward shorter jobs.
3) Score is based on SLO-budget risk:
       risk = (now - submit + predicted_remaining) / SLO_budget
   plus a queueing-risk term (LAS-like protection) and an aging term
   (starvation protection).
4) Urgency uses a piecewise (safe / danger / late) Knee function.

Configurable via environment-style attributes set by callers (see
`apply_config`). Default values match the design doc:

    theta = 0.7   gamma = 2.0
    c1 = 1.0      c2 = 6.0       c3 = 30.0
    w_size = 1.0  w_urg = 1.0    w_age = 0.1
    slo_target_seconds = 21600   (= 6h, calibrated for trace)
    risk_fn = "knee_quadratic"   options: linear, knee_quadratic, sigmoid
"""
from .scheduler_policy import SchedulingPolicy
import math
import os
import pandas as pd
from collections import defaultdict
from typing import Optional


class KneeSlo(SchedulingPolicy):
    DEFAULTS = {
        "theta": 0.7,
        "gamma": 2.0,
        "c1": 1.0,
        "c2": 6.0,
        "c3": 30.0,
        "w_size": 1.0,
        "w_urg": 1.0,
        "w_age": 0.1,
        "slo_target_seconds": 21600.0,  # 6h
        "age_threshold": 3600.0,
        "risk_fn": "knee_quadratic",
    }

    def __init__(self, args):
        self.metric_to_track = ["per_iter_time", "attained_service"]
        self.default_metric_value = [0, 0]

        # Apply defaults first
        for k, v in self.DEFAULTS.items():
            setattr(self, k, v)

        # Allow override from environment (used by parallel grid-search runs)
        for k in self.DEFAULTS:
            envkey = f"KNEE_{k.upper()}"
            if envkey in os.environ:
                raw = os.environ[envkey]
                try:
                    if k == "risk_fn":
                        setattr(self, k, raw)
                    else:
                        setattr(self, k, float(raw))
                except ValueError:
                    pass

        # Wall-clock sim time (updated externally each round)
        self.current_time = 0

        # Category-average profiled latency, keyed by job_class_id.
        self._cat_durations = defaultdict(list)
        self._cat_avg = {}
        self._global_avg = None

    # --- Hooks called by run_scheduler._sync_scheduler_state ------------------

    def update_profiled_latency(self, job):
        category = job.get("job_class_id", -1)
        actual = job.get("job_iteration_time", 1) * job.get("job_total_iteration", 1)
        self._cat_durations[category].append(actual)
        durs = self._cat_durations[category]
        self._cat_avg[category] = sum(durs) / len(durs)
        all_d = []
        for v in self._cat_durations.values():
            all_d.extend(v)
        self._global_avg = sum(all_d) / max(1, len(all_d))

    # --- Internal helpers -----------------------------------------------------

    def _predicted_total(self, job):
        cat = job.get("job_class_id", -1)
        if cat in self._cat_avg:
            return self._cat_avg[cat]
        if self._global_avg is not None:
            return self._global_avg
        # Cold-start fallback: coarse estimate.
        # Note: this leaks Oracle info on the FIRST job per category.
        # Subsequent jobs use the rolling category average above.
        return job.get("job_total_iteration", 300) * job.get("job_iteration_time", 1)

    def _predicted_remaining(self, job):
        total = self._predicted_total(job)
        attained = job.get("tracked_metrics", {}).get("attained_service", 0)
        return max(0.0, total - attained)

    def _urgency(self, risk):
        theta, gamma = self.theta, self.gamma
        c1, c2, c3 = self.c1, self.c2, self.c3

        if self.risk_fn == "linear":
            return c1 * risk

        if self.risk_fn == "sigmoid":
            # Smooth knee around theta.
            x = (risk - theta) * 6.0
            sig = 1.0 / (1.0 + math.exp(-x))
            base = c1 * risk
            return base + c2 * sig + (c3 * max(0.0, risk - 1.0))

        # default: knee_quadratic
        if risk < theta:
            return c1 * risk
        if risk < 1.0:
            x = (risk - theta) / max(1e-9, (1.0 - theta))
            return c1 * theta + c2 * (x ** gamma)
        return c1 * theta + c2 + c3 * (risk - 1.0)

    # --- Main entry point -----------------------------------------------------

    def _schedule_inner(self, job_dict, node_info, gpu_df):
        """The undecorated schedule body — subclasses can call this safely."""
        now = float(self.current_time)
        B = float(self.slo_target_seconds)
        global_avg = self._global_avg if self._global_avg else 300.0

        for jid, job in job_dict.items():
            submit = float(job.get("submit_time", 0))
            rem = self._predicted_remaining(job)
            pred_total = self._predicted_total(job)

            # Completion risk vs SLO budget.
            completion_risk = (now - submit + rem) / max(1.0, B)
            # Queueing risk: protects jobs that are waiting too long even if
            # their completion still fits the budget.
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

        sorted_job_order = sorted(
            job_dict.items(),
            key=lambda x: (x[1]["job_priority"], x[1]["slo_score"]),
        )

        return {
            "job_order": sorted_job_order,
            "run_all_jobs": False,
        }

    @SchedulingPolicy.copy_arguments
    def schedule(
        self,
        job_dict: dict,
        node_info: dict,
        gpu_df: pd.DataFrame,
        global_placement_policy: Optional[str] = None,
    ) -> dict:
        return self._schedule_inner(job_dict, node_info, gpu_df)
