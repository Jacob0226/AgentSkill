#!/usr/bin/env python3
"""Hold three exclusive GPU nodes on weekday mornings.

Each Monday–Friday at 04:00 Asia/Taipei, three 20h allocations start. They end
at 00:00 the next calendar day. Nothing new starts Saturday or Sunday.

Cluster clock is UTC. 04:00 Taipei is 20:00 UTC the previous day.
QoS wall clock caps a job at 24h. --begin only delays eligibility; it does
not reserve a node.

Idempotent. Re-running fills missing slots and replaces pending holds for a
target day that are not a 04:00 / 20h slot. Running jobs are left alone.
"""

import argparse
import os
import pwd
import re
import socket
import subprocess
import sys
from datetime import datetime, timedelta, timezone

TPE = timezone(timedelta(hours=8))

NAME_BASE = "Jacob_GLM_hold"
ACCOUNT = "amd-aifw-di"
QOS = "amd-aifw-di-qos"
PARTITION = "amd-spur"
GPUS = "8"
NODES = 3
START_HOUR = 4
WALL = timedelta(hours=20)
MIN_WALL = timedelta(minutes=30)

LOG = os.path.expanduser("~/hold-chain.log")
HOLD_BODY = (
    'echo "JobId: $SLURM_JOB_ID Node: $SLURM_NODELIST Started: $(date -Is)"; '
    'echo "Attach: srun --jobid=$SLURM_JOB_ID --pty bash"; '
    "sleep infinity"
)
NAME_RE = re.compile(
    rf"^{re.escape(NAME_BASE)}_(\d{{8}})(?:_([a-z])|([a-z]))?$"
)
ACTIVE = {"RUNNING", "PENDING", "CONFIGURING"}

DRY_RUN = False


def log(msg):
    host = socket.gethostname().split(".")[0]
    line = f"[{datetime.now(TPE):%F %H:%M:%S} TPE {host}] {msg}"
    print(line)
    if DRY_RUN:
        return
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def user_name():
    return os.environ.get("USER") or pwd.getpwuid(os.getuid()).pw_name


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def parse_timelimit(text):
    days, _, clock = text.partition("-")
    if not clock:
        days, clock = "0", days
    parts = [int(x) for x in clock.split(":")]
    while len(parts) < 3:
        parts.append(0)
    h, m, s = parts[:3]
    return timedelta(days=int(days), hours=h, minutes=m, seconds=s)


def fmt_timelimit(delta):
    total = max(0, int(delta.total_seconds()))
    d, rem = divmod(total, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    return f"{d}-{h:02d}:{m:02d}:{s:02d}" if d else f"{h:02d}:{m:02d}:{s:02d}"


def utc_stamp(dt):
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def at_start_hour(day):
    return day.astimezone(TPE).replace(
        hour=START_HOUR, minute=0, second=0, microsecond=0
    )


def week_monday(day):
    local = day.astimezone(TPE)
    monday = local - timedelta(days=local.isoweekday() - 1)
    return at_start_hour(monday)


def weekday_windows(now, through):
    """(start, end) in Taipei for each Mon–Fri 04:00 window still open."""
    if now.isoweekday() <= 5:
        first = week_monday(now)
    else:
        first = week_monday(now) + timedelta(days=7)
    last = at_start_hour(through) if through else first + timedelta(days=4)
    if last < first:
        last = first
    start = first
    while start <= last:
        end = start + WALL
        if start.isoweekday() <= 5 and end > now:
            yield start, end
        start += timedelta(days=1)


def parse_job(jid):
    detail = run(["scontrol", "show", "job", str(jid)])

    def field(key):
        match = re.search(rf"{key}=(\S+)", detail)
        return match.group(1) if match else None

    def ts(raw):
        if not raw or raw in ("N/A", "Unknown", "(null)", "None"):
            return None
        try:
            return datetime.fromisoformat(raw).replace(tzinfo=timezone.utc)
        except ValueError:
            return None

    name = field("JobName") or ""
    name_match = NAME_RE.match(name)
    begin_dt = None
    begin_raw = re.search(r"--begin=(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", detail)
    if begin_raw:
        begin_dt = datetime.fromisoformat(begin_raw.group(1)).replace(tzinfo=timezone.utc)

    tl_raw = field("TimeLimit")
    return {
        "id": int(jid),
        "name": name,
        "day": name_match.group(1) if name_match else None,
        "state": field("JobState") or "",
        "reason": field("Reason") or "",
        "end": ts(field("EndTime")),
        "timelimit": parse_timelimit(tl_raw) if tl_raw else WALL,
        "begin_dt": begin_dt,
    }


def list_hold_jobs():
    out = run(["squeue", "-u", user_name(), "-h", "-o", "%i|%j|%T"])
    jobs = []
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) != 3 or not parts[1].startswith(NAME_BASE):
            continue
        jobs.append(parse_job(parts[0]))
    jobs.sort(key=lambda job: job["id"])
    return jobs


def coverage_at(spans, moment):
    return sum(1 for start, end in spans if start <= moment < end)


def next_end(spans, moment, window_end):
    ends = [end for _, end in spans if moment < end <= window_end]
    return min(ends) if ends else window_end


def job_window_end(job):
    if job["begin_dt"] is not None:
        return job["begin_dt"] + job["timelimit"]
    if not job["day"]:
        return None
    start = datetime.strptime(job["day"], "%Y%m%d").replace(tzinfo=TPE, hour=START_HOUR)
    return (start + WALL).astimezone(timezone.utc)


def submit(walltime, name, begin=None):
    cmd = [
        "sbatch",
        "--parsable",
        f"--job-name={name}",
        f"--partition={PARTITION}",
        f"--account={ACCOUNT}",
        f"--qos={QOS}",
        f"--gpus={GPUS}",
        f"--time={fmt_timelimit(walltime)}",
        "--exclusive",
        f"--output={os.path.expanduser('~')}/hold-alloc-%j.out",
        f"--error={os.path.expanduser('~')}/hold-alloc-%j.err",
        f"--wrap={HOLD_BODY}",
    ]
    if begin is not None:
        cmd.insert(2, f"--begin={utc_stamp(begin)}")

    if DRY_RUN:
        log(
            f"DRY RUN would submit name={name} time={fmt_timelimit(walltime)} "
            f"begin={utc_stamp(begin) if begin else 'now'}"
        )
        return "dry"

    res = subprocess.run(cmd, capture_output=True, text=True)
    match = re.search(r"(\d+)", res.stdout)
    jid = match.group(1) if match else ""
    if res.returncode != 0 or not jid:
        log(f"sbatch failed: {res.stdout.strip()} {res.stderr.strip()}")
        return None
    log(
        f"submitted {jid} name={name} time={fmt_timelimit(walltime)} "
        f"begin={utc_stamp(begin) if begin else 'now'}"
    )
    return jid


def cancel_all(jobs, why):
    if not jobs:
        return True
    ids = [str(job["id"]) for job in jobs]
    log(why + " " + ", ".join(f"{job['id']} {job['name']}" for job in jobs))
    if DRY_RUN:
        return True
    res = subprocess.run(["scancel", *ids], capture_output=True, text=True)
    if res.returncode != 0:
        log(f"scancel failed: {res.stdout.strip()} {res.stderr.strip()}")
        return False
    return True


def release_held(jobs, now):
    """Release a hold whose window is still open. Cancel one whose window ended."""
    now_utc = now.astimezone(timezone.utc)
    release = []
    stale = []
    for job in jobs:
        if job["state"] != "PENDING" or job["reason"] != "JobHoldMaxRequeue":
            continue
        end = job_window_end(job)
        if end is not None and end > now_utc:
            release.append(job)
        else:
            stale.append(job)
    if not cancel_all(stale, "cancel stale JobHoldMaxRequeue"):
        return False
    for job in release:
        log(f"release {job['id']} {job['name']} JobHoldMaxRequeue")
        if DRY_RUN:
            continue
        res = subprocess.run(
            ["scontrol", "release", str(job["id"])], capture_output=True, text=True
        )
        if res.returncode != 0:
            log(f"release failed: {res.stdout.strip()} {res.stderr.strip()}")
            return False
    return True


def is_full_slot(job, start):
    """A future weekday slot: --begin at 04:00 Taipei and a 20h limit."""
    if job["begin_dt"] is None:
        return False
    if job["timelimit"] < WALL - timedelta(minutes=2):
        return False
    start_utc = start.astimezone(timezone.utc)
    return abs(job["begin_dt"] - start_utc) <= timedelta(minutes=5)


def fresh_name(day, used):
    for suffix in "abcdefgh":
        name = f"{NAME_BASE}_{day}_{suffix}"
        if name not in used:
            used.add(name)
            return name
    return None


def drop_bad_pending(jobs, start):
    """Pending jobs for this day that are not a 04:00 / 20h slot."""
    day = start.strftime("%Y%m%d")
    return [
        job
        for job in jobs
        if job["day"] == day and job["state"] == "PENDING" and not is_full_slot(job, start)
    ]


def ensure_future(jobs, start):
    day = start.strftime("%Y%m%d")
    mine = [job for job in jobs if job["day"] == day and job["state"] in ACTIVE]
    good = [job for job in mine if is_full_slot(job, start)]
    end = start + WALL
    log(
        f"future {start:%a %F %H:%M} → {end:%a %F %H:%M} TPE: "
        f"{len(good)}/{NODES} slots ({', '.join(str(job['id']) for job in good) or 'none'})"
    )
    used = {job["name"] for job in mine}
    for _ in range(NODES - len(good)):
        name = fresh_name(day, used)
        if name is None:
            log(f"no free job name for {day}")
            return
        if submit(WALL, name, begin=start) is None:
            return


def ensure_current(jobs, start, end, now):
    day = start.strftime("%Y%m%d")
    now_utc = now.astimezone(timezone.utc)
    end_utc = end.astimezone(timezone.utc)
    mine = [job for job in jobs if job["day"] == day and job["state"] in ACTIVE]
    spans = []
    for job in mine:
        if job["state"] == "RUNNING":
            job_end = job["end"] or end_utc
            if job["end"] is None:
                log(f"running {job['id']} has no EndTime; treating it as covering this window")
            spans.append((now_utc, min(job_end, end_utc)))
        elif job["begin_dt"] is not None:
            span_end = min(job["begin_dt"] + job["timelimit"], end_utc)
            spans.append((max(job["begin_dt"], now_utc), span_end))

    def projected_end(job):
        if job["end"]:
            return job["end"]
        if job["begin_dt"]:
            return job["begin_dt"] + job["timelimit"]
        return None

    log(
        f"current {start:%a %F %H:%M} → {end:%a %F %H:%M} TPE: "
        + ", ".join(
            f"{job['id']} {job['state']} until {projected_end(job).astimezone(TPE):%a %H:%M}"
            if projected_end(job)
            else f"{job['id']} {job['state']}"
            for job in mine
        )
        or "no jobs"
    )

    used = {job["name"] for job in mine}
    moment = now_utc
    while moment < end_utc:
        if end_utc - moment < MIN_WALL:
            break
        if coverage_at(spans, moment) >= NODES:
            nxt = next_end(spans, moment, end_utc)
            if nxt <= moment:
                break
            moment = nxt
            continue
        missing = NODES - coverage_at(spans, moment)
        wall = end_utc - moment
        begin = None if moment <= now_utc + timedelta(seconds=90) else moment
        for _ in range(missing):
            name = fresh_name(day, used)
            if name is None:
                log(f"no free job name for {day}")
                return
            if submit(wall, name, begin=begin) is None:
                return
            spans.append((moment, end_utc))
        return
    log("current window already has three nodes through midnight")


def main():
    global DRY_RUN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--through",
        metavar="YYYY-MM-DD",
        help="Last Taipei date to cover (inclusive). Default: Friday of this week, "
        "or next week when run on Saturday or Sunday.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    DRY_RUN = args.dry_run

    now = datetime.now(TPE)
    through = None
    if args.through:
        through = datetime.strptime(args.through, "%Y-%m-%d").replace(tzinfo=TPE)

    jobs = list_hold_jobs()
    windows = list(weekday_windows(now, through))
    log(
        "hold jobs: "
        + ", ".join(f"{job['id']} {job['name']} {job['state']}" for job in jobs)
    )
    if not release_held(jobs, now):
        return
    if not DRY_RUN:
        jobs = list_hold_jobs()

    bad = []
    for start, _end in windows:
        if now < start:
            bad.extend(drop_bad_pending(jobs, start))
    if not cancel_all(bad, "cancel pending that are not 04:00/20h"):
        return
    cancelled = {job["id"] for job in bad}
    jobs = [job for job in jobs if job["id"] not in cancelled]

    for start, end in windows:
        if now < start:
            ensure_future(jobs, start)
        else:
            ensure_current(jobs, start, end, now)


if __name__ == "__main__":
    main()
