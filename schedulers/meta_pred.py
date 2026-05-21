"""
Metadata-based prediction schedulers.

Reads `metadata_pred.json` (built by build_metadata_predictor.py) — a
linear regression over (predict_type, checkpoint_model, num_inference_steps,
num_images_per_prompt, num_lora) features.  Each job's predicted duration
is looked up by `job_id`.

This is a strictly non-Oracle predictor — the model was trained on jobs
BEFORE the tracked range (jobs 0-3000), and applied to the tracked
range (3000+) at scheduling time.

Two schedulers:
  - `MetaSrtf`: pure shortest-predicted-remaining first.
  - `MetaLasSlo`: LAS base + SLO emergency bucket using meta-predicted
    duration to define the safe / warning / late buckets.
"""
from .scheduler_policy import SchedulingPolicy
import os
import json
import pandas as pd
from typing import Optional


def _load_predictions():
    path = os.environ.get("META_PRED_FILE", "metadata_pred.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    preds = data.get("predictions", {})
    # Keys are like "application_003000" — also index by numeric id.
    out = {}
    for k, v in preds.items():
        out[k] = float(v)
        try:
            num = int(k.rsplit("_", 1)[-1])
            out[num] = float(v)
        except Exception:
            pass
    return out


_PRED_CACHE = None


def _predicted_dur(job):
    global _PRED_CACHE
    if _PRED_CACHE is None:
        _PRED_CACHE = _load_predictions()
    jid = job.get("job_id")
    if jid in _PRED_CACHE:
        return _PRED_CACHE[jid]
    # Fallback: category-mean style
    return job.get("job_total_iteration", 30) * job.get("job_iteration_time", 1)


class MetaSrtf(SchedulingPolicy):
    """SRTF using metadata-based predicted remaining."""

    def __init__(self, args):
        self.metric_to_track = ["per_iter_time", "attained_service"]
        self.default_metric_value = [0, 0]
        self.current_time = 0

    @SchedulingPolicy.copy_arguments
    def schedule(
        self,
        job_dict: dict,
        node_info: dict,
        gpu_df: pd.DataFrame,
        global_placement_policy: Optional[str] = None,
    ) -> dict:
        for jid, job in job_dict.items():
            total = _predicted_dur(job)
            attained = job.get("tracked_metrics", {}).get("attained_service", 0)
            rem = max(0.0, total - attained)
            job["meta_pred_score"] = rem

        sorted_order = sorted(
            job_dict.items(),
            key=lambda x: (x[1]["job_priority"], x[1]["meta_pred_score"]),
        )
        return {"job_order": sorted_order, "run_all_jobs": False}


class MetaLasSlo(SchedulingPolicy):
    """LAS base + SLO emergency bucket; SLO target can be flat OR
    metadata-scaled (predicted_dur × multiplier)."""

    def __init__(self, args):
        self.metric_to_track = ["per_iter_time", "attained_service"]
        self.default_metric_value = [0, 0]
        self.current_time = 0
        self.slo_target = float(os.environ.get("META_SLO_TARGET", "60"))
        # If META_SLO_MULT is set, SLO becomes pred_dur * multiplier per-job.
        self.slo_multiplier = float(os.environ.get("META_SLO_MULT", "0"))
        self.theta = float(os.environ.get("META_SLO_THETA", "0.7"))

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
            attained = job.get("tracked_metrics", {}).get("attained_service", 0)
            submit = float(job.get("submit_time", 0))
            wait = now - submit

            # Per-job SLO budget — metadata-scaled if multiplier set.
            if self.slo_multiplier > 0:
                B = max(5.0, self.slo_multiplier * _predicted_dur(job))
            else:
                B = self.slo_target

            warning = self.theta * B
            if wait >= B:
                bucket = 0
                secondary = -wait
            elif wait >= warning:
                bucket = 1
                secondary = attained
            else:
                bucket = 2
                secondary = attained  # pure LAS

            job["meta_las_slo_score"] = (bucket, secondary)

        sorted_order = sorted(
            job_dict.items(),
            key=lambda x: (x[1]["job_priority"], x[1]["meta_las_slo_score"]),
        )
        return {"job_order": sorted_order, "run_all_jobs": False}
