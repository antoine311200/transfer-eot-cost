"""Optional Weights & Biases integration for DualWCFM.

Falls back gracefully to a no-op stub if wandb is not installed or the user
did not supply a project name.  All callers should therefore be written as::

    run = WandbRun.init(project="my-project", name="exp-name", config={...})
    run.log({"loss": 0.1}, step=42)
    run.finish()

and nothing will break if wandb is unavailable.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

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
        kwargs = {"step": step} if step is not None else {}
        self._run.log(metrics, **kwargs)

    def log_image(self, key: str, images, step: Optional[int] = None) -> None:
        try:
            import wandb
            kwargs = {"step": step} if step is not None else {}
            self._run.log({key: [wandb.Image(img) for img in images]}, **kwargs)
        except Exception:
            pass

    def summary(self, metrics: Dict[str, Any]) -> None:
        for k, v in metrics.items():
            self._run.summary[k] = v

    def finish(self) -> None:
        self._run.finish()

    def __bool__(self) -> bool:
        return True


def init(
    project: Optional[str] = None,
    name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> "_NoOpRun | _WandbRun":
    """Initialise a W&B run, or return a no-op stub if wandb is unavailable.

    Parameters
    ----------
    project:
        W&B project name.  Pass *None* to disable logging entirely.
    name:
        W&B run name.  Defaults to wandb auto-generated names.
    config:
        Flat dict of hyper-parameters to log as run configuration.

    Returns
    -------
    A :class:`_WandbRun` or :class:`_NoOpRun` with the same interface.
    """
    if project is None:
        return _NoOpRun()

    try:
        import wandb
        run = wandb.init(
            project = project,
            name    = name,
            config  = config or {},
            reinit  = True,
        )
        logger.info("  W&B run initialised: %s / %s", project, run.name)
        return _WandbRun(run)
    except Exception as exc:
        logger.warning("  W&B init failed (%s) — logging disabled.", exc)
        return _NoOpRun()
