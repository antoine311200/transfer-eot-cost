"""Optional Weights & Biases integration for TACO.

Falls back gracefully to a no-op stub if wandb is not installed or the user
did not supply a project name.  All callers should therefore be written as::

    run = WandbRun.init(project="my-project", name="exp-name", config={...})
    run.log({"loss": 0.1}, step=42)
    run.finish()

and nothing will break if wandb is unavailable.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("dualwcfm")


class _NoOpRun:
    """Stub returned when W&B is disabled.  All methods are no-ops."""

    enabled: bool = False

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        pass

    def log_image(self, key: str, images, step: Optional[int] = None) -> None:
        pass

    def summary(self, metrics: Dict[str, Any]) -> None:
        pass

    def finish(self) -> None:
        pass

    def __bool__(self) -> bool:
        return False


class _WandbRun:
    """Thin wrapper around a live wandb Run."""

    enabled: bool = True

    def __init__(self, run) -> None:
        self._run = run

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        try:
            kwargs = {"step": step} if step is not None else {}
            self._run.log(metrics, **kwargs)
        except Exception as exc:
            logger.warning("  W&B log failed (%s) — skipping.", exc)

    def log_image(self, key: str, images, step: Optional[int] = None) -> None:
        try:
            import wandb
            kwargs = {"step": step} if step is not None else {}
            self._run.log({key: [wandb.Image(img) for img in images]}, **kwargs)
        except Exception:
            pass

    def summary(self, metrics: Dict[str, Any]) -> None:
        try:
            for k, v in metrics.items():
                self._run.summary[k] = v
        except Exception as exc:
            logger.warning("  W&B summary failed (%s) — skipping.", exc)

    def finish(self) -> None:
        try:
            self._run.finish()
        except Exception:
            pass

    def __bool__(self) -> bool:
        return True


def init(
    project: Optional[str] = None,
    name: Optional[str] = None,
    entity: Optional[str] = None,
    group: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
    mode: Optional[str] = None,
) -> "_NoOpRun | _WandbRun":
    """Initialise a W&B run, or return a no-op stub if wandb is unavailable.

    Parameters
    ----------
    project:
        W&B project name (e.g. ``"Data-Based EOT"``).  Pass *None* to disable.
    name:
        Run display name — becomes the experiment name (e.g. ``"mnist_simple"``).
    entity:
        W&B entity (username or team, e.g. ``"antoine311200"``).  When *None*
        wandb uses the default entity from ``wandb login``.
    group:
        Group name for clustering related runs in the W&B UI (useful for
        ablation suites, e.g. ``"ablation_particle_ratio"``).
    config:
        Flat dict of hyper-parameters logged as run configuration.
    tags:
        List of string tags attached to the run (e.g. ``["mnist_simple"]``).
        Tags are searchable and filterable in the W&B UI.

    Returns
    -------
    A :class:`_WandbRun` or :class:`_NoOpRun` with the same interface.
    """
    if project is None:
        return _NoOpRun()

    # WANDB_MODE env var takes precedence; explicit mode= overrides only when set.
    import os
    resolved_mode = mode or os.environ.get("WANDB_MODE", "online")

    try:
        import wandb
        run = wandb.init(
            project  = project,
            name     = name,
            entity   = entity,
            group    = group,
            config   = config or {},
            tags     = tags or [],
            reinit   = True,
            mode     = resolved_mode,
            settings = wandb.Settings(init_timeout=120),
        )
        # wandb sometimes ignores name= when reinit=True; force it explicitly.
        if name and run.name != name:
            run.name = name
            run.save()
        logger.info(
            "  W&B run initialised (mode=%s): %s/%s — run: %s",
            resolved_mode, entity or "<default>", project, run.name,
        )
        return _WandbRun(run)
    except Exception as exc:
        logger.warning("  W&B init failed (%s) — logging disabled.", exc)
        return _NoOpRun()
