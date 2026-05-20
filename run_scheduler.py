"""
Generic scheduler runner that dynamically selects the scheduling policy
based on command-line argument --scheduler-name.

Usage:
  python run_scheduler.py --simulate --load 1 --exp-prefix test --scheduler-name Fifo
  python run_scheduler.py --simulate --load 1 --exp-prefix test --scheduler-name Srtf
  python run_scheduler.py --simulate --load 1 --exp-prefix test --scheduler-name Las
"""
import os
import sys
import argparse

import schedulers
from placement import placement
import admission_control
from blox import ClusterState, JobState, BloxManager
import blox.utils as utils

SCHEDULER_MAP = {
    "Fifo": schedulers.Fifo,
    "Las": schedulers.Las,
    "Srtf": schedulers.Srtf,
    "SloScoring": schedulers.SloScoring,
}


def parse_args(parser):
    parser.add_argument("--scheduler", default="Fifo", type=str)
    parser.add_argument("--node-manager-port", default=50052, type=int)
    parser.add_argument("--central-scheduler-port", default=50051, type=int)
    parser.add_argument("--simulator-rpc-port", default=50050, type=int)
    parser.add_argument("--scheduler-name", default="Fifo", type=str)
    parser.add_argument("--placement-name", default="Las", type=str)
    parser.add_argument("--acceptance-policy", default="accept_all", type=str)
    parser.add_argument("--plot", action="store_true", default=False)
    parser.add_argument("--exp-prefix", type=str)
    parser.add_argument("--load", type=int)
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--round-duration", type=int, default=300)
    parser.add_argument("--start-id-track", type=int, default=3000)
    parser.add_argument("--stop-id-track", type=int, default=4000)
    return parser.parse_args()


def main(args):
    sched_cls = SCHEDULER_MAP.get(args.scheduler_name)
    if sched_cls is None:
        print(f"Unknown scheduler: {args.scheduler_name}")
        print(f"Available: {list(SCHEDULER_MAP.keys())}")
        sys.exit(1)

    scheduling_policy = sched_cls(args)
    placement_policy = placement.JobPlacement(args)
    admission_policy = admission_control.acceptAll(args)

    if not args.simulate:
        print("Only simulation mode is supported in this script.")
        sys.exit(1)

    blox_instance = BloxManager(args)
    new_config = blox_instance.rmserver.get_new_sim_config()
    print(f"New config {new_config}")

    if args.scheduler_name == "":
        blox_instance.terminate_server()
        sys.exit()

    blox_instance.scheduler_name = new_config["scheduler"]
    blox_instance.load = new_config["load"]
    args.scheduler_name = new_config["scheduler"]
    args.load = new_config["load"]
    args.start_id_track = new_config["start_id_track"]
    args.stop_id_track = new_config["stop_id_track"]

    print(f"Running Scheduler {args.scheduler_name}\nLoad {args.load}")

    blox_instance.reset(args)
    cluster_state = ClusterState(args)
    job_state = JobState(args)
    os.environ["sched_policy"] = args.scheduler_name
    os.environ["sched_load"] = str(args.load)

    while True:
        if blox_instance.terminate:
            blox_instance.terminate_server()
            print(f"Terminate: {args.scheduler_name}")
            break
        blox_instance.update_cluster(cluster_state)
        blox_instance.update_metrics(cluster_state, job_state)
        new_jobs = blox_instance.pop_wait_queue(args.simulate)
        accepted_jobs = admission_policy.accept(new_jobs, cluster_state, job_state)
        job_state.add_new_jobs(accepted_jobs)
        new_job_schedule = scheduling_policy.schedule(job_state, cluster_state)
        utils.prune_jobs(job_state, cluster_state, blox_instance)
        new_job_schedule = scheduling_policy.schedule(job_state, cluster_state)
        to_suspend, to_launch = placement_policy.place(
            job_state, cluster_state, new_job_schedule
        )
        utils.collect_custom_metrics(
            job_state, cluster_state, {"num_preemptions": len(to_suspend)}
        )
        utils.collect_cluster_job_metrics(job_state, cluster_state)
        utils.track_finished_jobs(job_state, cluster_state, blox_instance)
        blox_instance.exec_jobs(to_launch, to_suspend, cluster_state, job_state)
        args.round_duration = 300
        job_state.time += args.round_duration
        cluster_state.time += args.round_duration
        blox_instance.time += args.round_duration


if __name__ == "__main__":
    args = parse_args(
        argparse.ArgumentParser(description="Generic Blox Scheduler Runner")
    )
    main(args)
