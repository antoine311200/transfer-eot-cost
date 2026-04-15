"""Rename all W&B runs in a project based on their logged config.

For each run the new name is built from config keys that were always logged:
  {experiment}__{method}__s{sigma}__pr{particle_ratio}__it{iterations}

Runs that already have a non-random name (i.e. contain the experiment name)
are skipped.

Usage
-----
    python rename_wandb_runs.py --entity antoine311200 --project "Data-Based EOT"

Dry-run (print proposed renames without applying):
    python rename_wandb_runs.py --entity antoine311200 --project "Data-Based EOT" --dry-run
"""
import argparse
import re


def _fmt_float(v) -> str:
    try:
        f = float(v)
        s = f"{f:.0e}".replace("+0", "").replace("-0", "-")
        return s
    except (TypeError, ValueError):
        return str(v)


def _build_name(config: dict) -> str:
    """Construct a human-readable run name from the logged config."""
    parts = []

    experiment = config.get("experiment") or config.get("base_experiment")
    if experiment:
        parts.append(str(experiment))

    method = config.get("method")
    if method:
        parts.append(str(method))

    sigma = config.get("sigma")
    if sigma is not None:
        parts.append(f"s{_fmt_float(sigma)}")

    pr = config.get("particle_ratio")
    if pr is not None:
        parts.append(f"pr{_fmt_float(pr)}")

    iters = config.get("iterations")
    if iters is not None:
        parts.append(f"it{iters}")

    return "__".join(parts) if parts else ""


def _looks_random(name: str, known_experiments: set) -> bool:
    """Return True if the run name looks like a wandb auto-generated name."""
    if not name:
        return True
    # If it already contains a known experiment substring, it was set by us.
    for exp in known_experiments:
        if exp in name:
            return False
    # wandb random names look like "adjective-noun-42" or contain "sweep"
    if re.fullmatch(r"[a-z]+-[a-z]+-\d+", name):
        return True
    if "sweep" in name.lower():
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--entity",  required=True, help="W&B entity (username or team)")
    parser.add_argument("--project", required=True, help="W&B project name")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print proposed renames without applying them")
    parser.add_argument("--force",   action="store_true",
                        help="Rename ALL runs, even those that already have a custom name")
    args = parser.parse_args()

    try:
        import wandb
    except ImportError:
        raise SystemExit("wandb is not installed. Run: pip install wandb")

    api = wandb.Api()
    runs = list(api.runs(f"{args.entity}/{args.project}"))
    print(f"Found {len(runs)} runs in {args.entity}/{args.project}")

    # Collect all known experiment names from configs so we can detect random names
    known_experiments = set()
    for run in runs:
        exp = run.config.get("experiment") or run.config.get("base_experiment")
        if exp:
            known_experiments.add(str(exp))

    renamed = skipped = already_ok = 0

    for run in runs:
        current_name = run.name or ""
        new_name = _build_name(run.config)

        if not new_name:
            print(f"  SKIP (no config)  {run.id}  '{current_name}'")
            skipped += 1
            continue

        is_random = _looks_random(current_name, known_experiments)

        if not args.force and not is_random:
            print(f"  OK (kept)         {run.id}  '{current_name}'")
            already_ok += 1
            continue

        if args.dry_run:
            print(f"  WOULD RENAME      {run.id}  '{current_name}'  →  '{new_name}'")
        else:
            run.name = new_name
            run.save()
            print(f"  RENAMED           {run.id}  '{current_name}'  →  '{new_name}'")
        renamed += 1

    print()
    print(f"Summary: {renamed} renamed, {already_ok} already ok, {skipped} skipped (no config)")
    if args.dry_run:
        print("(dry-run — no changes applied; re-run without --dry-run to apply)")


if __name__ == "__main__":
    main()
