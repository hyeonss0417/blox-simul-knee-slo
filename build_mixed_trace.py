"""
Build a mixed-inference-+-training trace from the Alibaba 2026 inference
trace.  Goal: dramatically increase CoV (job duration heterogeneity).

Mixing recipe:
  - 80% jobs: keep original inference duration (5-145s)
  - 20% jobs: replace duration with synthetic 'training-like' long job
              (10min ~ 2h, lognormal so heavy-tailed)

Output: cluster_job_log_mixed.json (Philly Trace format compatible).
"""
import json, datetime, random, math, os, sys

SRC = "cluster_job_log"
OUT = "cluster_job_log_mixed"
TRAINING_FRACTION = 0.20
SEED = 42

random.seed(SEED)


def fmt(t): return datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S")
def fmt2str(t): return t.strftime("%Y-%m-%d %H:%M:%S")


def sample_training_duration():
    """Lognormal training-like duration: mean ~30 min, heavy right tail."""
    # mean of log = log(1800), std = 0.8 → median 1800s, P95 ~6300s
    mu, sigma = math.log(1800), 0.8
    d = random.lognormvariate(mu, sigma)
    return max(600, min(7200, d))   # clamp to [10min, 2h]


def main():
    print(f"Reading {SRC} ...")
    data = json.load(open(SRC))
    print(f"  {len(data)} jobs")

    n_training = 0
    durations_inf, durations_tr = [], []

    for j in data:
        if not j.get("attempts"):
            continue
        a = j["attempts"][0]
        try:
            orig_dur = (fmt(a["end_time"]) - fmt(a["start_time"])).total_seconds()
        except Exception:
            continue
        if orig_dur <= 0:
            continue

        is_training = random.random() < TRAINING_FRACTION
        if is_training:
            new_dur = sample_training_duration()
            durations_tr.append(new_dur)
            n_training += 1
            # Override duration: keep start_time, recompute end_time
            start = fmt(a["start_time"])
            new_end = start + datetime.timedelta(seconds=new_dur)
            a["end_time"] = fmt2str(new_end)
            # Mark in metadata
            j["predict_type"] = "TRAINING_SYNTHETIC"
            j["num_inference_steps"] = -1
            # Note: we keep checkpoint_model — could be the source signal,
            # but predictor was trained on inference only so it'll predict
            # ~average inference time for these → poor MetaSrtf for them.
        else:
            durations_inf.append(orig_dur)

    print(f"  Training jobs injected: {n_training} ({n_training/len(data)*100:.1f}%)")
    import statistics
    if durations_inf:
        print(f"  Inference (kept): mean={statistics.mean(durations_inf):.1f}s, "
              f"median={statistics.median(durations_inf):.0f}s")
    if durations_tr:
        print(f"  Training (synthetic): mean={statistics.mean(durations_tr):.0f}s, "
              f"median={statistics.median(durations_tr):.0f}s, "
              f"max={max(durations_tr):.0f}s")

    # Mixed stats
    all_d = durations_inf + durations_tr
    mean = statistics.mean(all_d)
    std = statistics.stdev(all_d)
    print(f"\n  MIXED workload stats:")
    print(f"    n={len(all_d)}, mean={mean:.1f}s, std={std:.1f}s")
    print(f"    **CoV = {std/mean:.3f}** (vs 0.73 for pure inference)")
    print(f"    Theoretical SJF gain ≈ {(std/mean)**2/2*100:.1f}%")

    with open(OUT, "w") as f:
        json.dump(data, f)
    print(f"\nSaved {OUT}  ({os.path.getsize(OUT)/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
