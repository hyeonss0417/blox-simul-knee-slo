"""
Convert Alibaba 2026 GenAI (Stable Diffusion) trace to Blox Philly Trace format.

Input:  trace-data/lora_request_trace.csv
Output: cluster_job_log (JSON, Philly Trace format)

Mapping:
  gmt_create           -> submitted_time, attempts[].start_time
  exec_time_seconds    -> duration (start_time + duration = end_time)
  predict_type         -> vc (TXT_2_IMG, IMG_2_IMG, INPAINTING)
  num_images_per_prompt -> GPU demand heuristic
  groupId              -> user
  predict_status       -> status (SUCCEED->Pass, FAILED->Failed)
"""
import json
import datetime
import pandas as pd
import argparse


GPU_DEMAND_MAP = {
    1.0: 1,
    2.0: 1,
    3.0: 2,
    4.0: 2,
    5.0: 4,
    8.0: 4,
}

STATUS_MAP = {
    "SUCCEED": "Pass",
    "FAILED": "Failed",
    "PENDING": "Killed",
    "PROCESSING": "Killed",
}


def convert(input_path, output_path, max_jobs=None):
    df = pd.read_csv(input_path)
    df = df.dropna(subset=["gmt_create", "exec_time_seconds"])
    df = df[df["exec_time_seconds"] > 0]
    df = df.sort_values("gmt_create").reset_index(drop=True)

    if max_jobs:
        df = df.head(max_jobs)

    jobs = []
    for i, row in df.iterrows():
        submitted = row["gmt_create"]
        duration_sec = row["exec_time_seconds"]

        start_dt = datetime.datetime.strptime(submitted, "%Y-%m-%d %H:%M:%S")
        queue_delay = 5.0
        start_time = start_dt + datetime.timedelta(seconds=queue_delay)
        end_time = start_time + datetime.timedelta(seconds=duration_sec)

        gpu_demand = GPU_DEMAND_MAP.get(row.get("num_images_per_prompt", 1.0), 1)
        gpus = [f"gpu{g}" for g in range(gpu_demand)]

        job = {
            "status": STATUS_MAP.get(row.get("predict_status", "SUCCEED"), "Pass"),
            "vc": str(row.get("predict_type", "TXT_2_IMG")),
            "jobid": f"application_{i:06d}",
            "attempts": [
                {
                    "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "detail": [{"ip": f"m{i % 64 + 1}", "gpus": gpus}],
                }
            ],
            "submitted_time": submitted,
            "user": str(row.get("groupId", "unknown")),
            "predict_type": str(row.get("predict_type", "")),
            "num_inference_steps": int(row.get("num_inference_steps", 30)),
            "num_images_per_prompt": int(row.get("num_images_per_prompt", 1)),
            "prompt_length": float(row.get("prompt_length", 0)),
            "exec_time_seconds": float(duration_sec),
            "checkpoint_model": str(row.get("checkpoint_model_version_id", "")),
            "num_lora": int(row.get("num_lora", 0)),
        }
        jobs.append(job)

    with open(output_path, "w") as f:
        json.dump(jobs, f, indent=2)

    print(f"Converted {len(jobs)} jobs -> {output_path}")
    print(f"  predict_type distribution: {df['predict_type'].value_counts().to_dict()}")
    print(f"  exec_time: mean={df['exec_time_seconds'].mean():.1f}s, "
          f"median={df['exec_time_seconds'].median():.1f}s")
    print(f"  GPU demand distribution: "
          f"{df['num_images_per_prompt'].map(GPU_DEMAND_MAP).fillna(1).astype(int).value_counts().to_dict()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="trace-data/lora_request_trace.csv")
    parser.add_argument("--output", default="cluster_job_log")
    parser.add_argument("--max-jobs", type=int, default=None)
    args = parser.parse_args()
    convert(args.input, args.output, args.max_jobs)
