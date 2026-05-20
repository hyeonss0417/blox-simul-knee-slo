"""
Generate a synthetic cluster_job_log in the same JSON format as the
Microsoft Philly Trace for use with the Blox simulator.
"""
import json
import random
import datetime

random.seed(42)

NUM_JOBS = 500
BASE_TIME = datetime.datetime(2017, 10, 1, 0, 0, 0)
USERS = [f"user_{i:04d}" for i in range(20)]
VCS = [f"vc_{i:02d}" for i in range(5)]
MACHINES = [f"m{i}" for i in range(1, 65)]

jobs = []
current_time = BASE_TIME

for i in range(NUM_JOBS):
    # Random inter-arrival: 10s ~ 600s
    gap = random.uniform(10, 600)
    current_time += datetime.timedelta(seconds=gap)
    submitted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

    # Queue delay: 0 ~ 300s
    queue_delay = random.uniform(0, 300)
    start = current_time + datetime.timedelta(seconds=queue_delay)

    # Duration: 1min ~ 10hrs (matches Philly trace distribution roughly)
    duration = random.uniform(60, 36000)
    end = start + datetime.timedelta(seconds=duration)

    # GPU demand: 1, 2, 4, or 8
    gpu_demand = random.choice([1, 1, 1, 1, 2, 2, 4, 8])
    machine = random.choice(MACHINES)
    gpus = [f"gpu{g}" for g in range(gpu_demand)]

    job = {
        "status": random.choice(["Pass", "Pass", "Pass", "Killed", "Failed"]),
        "vc": random.choice(VCS),
        "jobid": f"application_{i:06d}",
        "attempts": [
            {
                "start_time": start.strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": end.strftime("%Y-%m-%d %H:%M:%S"),
                "detail": [{"ip": machine, "gpus": gpus}],
            }
        ],
        "submitted_time": submitted_time,
        "user": random.choice(USERS),
    }
    jobs.append(job)

with open("cluster_job_log", "w") as f:
    json.dump(jobs, f, indent=2)

print(f"Generated {NUM_JOBS} synthetic jobs -> cluster_job_log")
