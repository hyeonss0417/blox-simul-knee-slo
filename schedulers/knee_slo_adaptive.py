"""
Knee-SLO with adaptive theta based on system load.

Static theta=0.7 might be too eager under low load or too lax under high
load. We let theta drift between [0.4, 0.8] based on queue depth:

    pressure = n_active_jobs / total_GPUs
    theta    = clamp(0.8 - 0.4 * pressure, 0.4, 0.8)

So under high pressure (queue building up), theta drops — we become
aggressive earlier.  Under low pressure, theta stays at 0.8 — SLO budget
absorbs spikes.
"""
from .knee_slo import KneeSlo
from .scheduler_policy import SchedulingPolicy
import pandas as pd
from typing import Optional


class KneeSloAdaptive(KneeSlo):
    @SchedulingPolicy.copy_arguments
    def schedule(
        self,
        job_dict: dict,
        node_info: dict,
        gpu_df: pd.DataFrame,
        global_placement_policy: Optional[str] = None,
    ) -> dict:
        total_gpus = max(1, len(gpu_df))
        n_jobs = len(job_dict)
        pressure = n_jobs / total_gpus
        self.theta = max(0.4, min(0.8, 0.8 - 0.4 * pressure))
        return self._schedule_inner(job_dict, node_info, gpu_df)
