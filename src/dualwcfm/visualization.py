from __future__ import annotations

from typing import Optional, Tuple, TYPE_CHECKING, Callable

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import MultipleLocator

from dualwcfm.config import VisualizeConfig
from dualwcfm.utils import is_image_tensor
from dualwcfm.strategy import BidirectionalStrategy

if TYPE_CHECKING:
    from dualwcfm.trainer import Trainer


class Visualizer:
    def __init__(self, config: VisualizeConfig, transport_map: Callable):
        self.config = config
        self.transport_map = transport_map
        self.dim = 2

    # ------------------------------------------------------------------
    # Iteration-level display (called from Trainer.train loop)
    # ------------------------------------------------------------------

    def display(self, epoch: int, trainer: "Trainer"):
        if epoch % self.config.show_each == 0:
            x, y = trainer.current_coupling.coupling_sample(self.config.n_trajectories)
            x_target = trainer.new_source.sample(self.config.n_trajectories)
            y_target = trainer.new_target.sample(self.config.n_trajectories)
            x, y = x.to(trainer.device), y.to(trainer.device)
            x_target, y_target = x_target.to(trainer.device), y_target.to(trainer.device)

            start_points = y if isinstance(trainer.strategy, BidirectionalStrategy) and epoch % 2 == 1 else x
            target_points = x_target if isinstance(trainer.strategy, BidirectionalStrategy) and epoch % 2 == 1 else y_target
            traj = trainer.flow_matching.evolve_trajectory(
                start_points,
                trainer.strategy.get_flow(epoch, trainer.device)[0],
                num_steps=25,
                sigma=trainer._sigma,
                extend=True,
            )  # (26, N, 2D) — first D dims are the evolved part
            traj = traj[:, :, : self.dim]  # (26, N, D)
            s = trainer.scale
            stride = self.config.trajectory_stride
            traj_2d = (traj * s).cpu() if stride > 0 else None
            self._display_evolution(
                (start_points * s).cpu(), (traj[-1] * s).cpu(), (target_points * s).cpu(),
                trajectories=traj_2d, trajectory_stride=stride,
            )
            plt.show()

    def save_fig(self, fig: plt.Figure, subfolder: str, name: str):
        """Save or display a figure. Override in logging visualizers to save to disk."""
        plt.show()
        plt.close(fig)

    # ------------------------------------------------------------------
    # Pretrain lifecycle hooks
    # ------------------------------------------------------------------

    def at_pretrain_start(self, trainer: "Trainer", ctx: dict) -> None:
        pass

    def during_pretrain(self, epoch: int, trainer: "Trainer", ctx: dict) -> None:
        """Called each epoch during pretrain.

        ctx keys:
          phase == "bridges":  mu_model, nu_model, phase_idx (int), force (bool)
          phase == "flow":     model, backward (bool)
        """
        phase = ctx.get("phase", "bridges")
        if phase == "bridges":
            if self.config.show_each_marginal <= 0:
                return
            force = ctx.get("force", False)
            if not force and (epoch + 1) % self.config.show_each_marginal != 0:
                return
            mu_model = ctx["mu_model"]
            nu_model = ctx["nu_model"]
            phase_idx = ctx.get("phase_idx", 0)
            n = min(self.config.n_trajectories, 500)
            s = trainer.scale
            stride = self.config.trajectory_stride
            for model, source_dist, target_dist, label in [
                (mu_model, trainer.mu_distribution, trainer.new_source, "Mu"),
                (nu_model, trainer.nu_distribution, trainer.new_target, "Nu"),
            ]:
                start = source_dist.sample(n).to(trainer.device)
                with torch.no_grad():
                    traj = trainer.flow_matching.evolve_trajectory(
                        start, model, num_steps=50, sigma=trainer._sigma, extend=False
                    )
                end = traj[-1]
                target = target_dist.sample(n).to(trainer.device)
                traj_2d = (traj[:, :, :self.dim] * s).cpu() if stride > 0 else None
                self._display_evolution(
                    (start * s).cpu(), (end * s).cpu(), (target * s).cpu(),
                    trajectories=traj_2d, trajectory_stride=stride,
                )
                phase_tag = "Rectified " if phase_idx > 0 else ""
                plt.suptitle(f"{label} {phase_tag}Marginal Flow — Epoch {epoch} Phase {phase_idx}")
                rectified_suffix = "_rectified" if phase_idx > 0 else ""
                self.save_fig(
                    plt.gcf(), "pretrain",
                    f"{label.lower()}_bridge{rectified_suffix}_{epoch:05d}_p{phase_idx}.png",
                )

        elif phase == "flow":
            if self.config.show_each_pretrain <= 0:
                return
            if (epoch + 1) % self.config.show_each_pretrain != 0:
                return
            model = ctx["model"]
            backward = ctx.get("backward", False)
            n = min(self.config.n_trajectories, 500)
            s = trainer.scale
            x0, x1 = trainer.initial_coupling.coupling_sample(n)
            x0, x1 = x0.to(trainer.device), x1.to(trainer.device)
            if backward:
                x0, x1 = x1, x0
            with torch.no_grad():
                end = trainer.flow_matching.evolve(
                    x0, model, num_steps=50, sigma=trainer._sigma, extend=True
                )
                end = end[:, :self.dim]
            direction = "Backward" if backward else "Forward"
            self._display_evolution((x0 * s).cpu(), (end * s).cpu(), (x1 * s).cpu())
            plt.suptitle(f"Pretrain {direction} Flow — Epoch {epoch}")
            self.save_fig(
                plt.gcf(), "pretrain",
                f"pretrain_{'bwd' if backward else 'fwd'}_{epoch:04d}.png",
            )

    def at_pretrain_end(self, trainer: "Trainer", ctx: dict) -> None:
        """Visualize final bridge quality (mu and nu bridges)."""
        pass
        # samples = 2500
        # s = trainer.scale
        # for model_attr, src_attr, tgt_attr, title, fname in [
        #     ("mu_bridge", "mu_distribution", "new_source",
        #      "Mu bridge: source distribution flow", "mu_bridge_final.png"),
        #     ("nu_bridge", "nu_distribution", "new_target",
        #      "Nu bridge: target distribution flow", "nu_bridge_final.png"),
        # ]:
        #     model = getattr(trainer.strategy, model_attr)
        #     start = getattr(trainer, src_attr).sample(samples).to(trainer.device)
        #     model.eval()
        #     with torch.no_grad():
        #         end = trainer.flow_matching.evolve(
        #             start, model, num_steps=trainer.config.advance_steps, extend=False
        #         )
        #     target = getattr(trainer, tgt_attr).sample(samples).to(trainer.device)
        #     self._display_evolution(
        #         (start * s).cpu(), (end * s).cpu(), (target * s).cpu()
        #     )
        #     plt.suptitle(title)
        #     self.save_fig(plt.gcf(), "marginals", fname)

    def at_flow_pretrain_end(self, trainer: "Trainer", ctx: dict) -> None:
        """Visualize the pretrained forward (and backward) flows on the new marginals.

        Called once after the initial flow models are trained on the reference
        coupling.  Shows how poorly (or well) the pretrained model handles the
        *new* marginals **before** any dual adaptation begins — a key diagnostic
        for the paper comparing the pretrained baseline against the adapted result.

        Forward:  new_source → pretrained_forward → compare vs new_target
        Backward: new_target → pretrained_backward → compare vs new_source

        Saves to ``pretrain/pretrained_fwd_new_marginals.png``
        (and ``pretrain/pretrained_bwd_new_marginals.png`` if bidirectional).
        """
        strategy = trainer.strategy
        fwd_model = getattr(strategy, "initial_forward_model", None)
        if fwd_model is None:
            return

        n = min(self.config.n_trajectories, 500)
        device = trainer.device
        s = trainer.scale
        dim = trainer.strategy.config.dimension

        x_new = trainer.new_source.sample(n).to(device)
        fwd_model.eval()
        with torch.no_grad():
            evolved = trainer.flow_matching.evolve(
                x_new, fwd_model, num_steps=100, sigma=0.0, extend=True
            )
            y_hat = evolved[:, :dim]
        y_true = trainer.new_target.sample(n).to(device)

        map_err_str = ""
        if self.transport_map is not None:
            with torch.no_grad():
                err = (y_hat - self.transport_map(x_new)).pow(2).sum(-1).mean().sqrt()
            map_err_str = f"  (map err = {err.item() * s:.4f})"

        self._display_evolution(
            (x_new * s).cpu(), (y_hat * s).cpu(), (y_true * s).cpu(),
        )
        plt.suptitle(
            f"Pretrained forward flow on new marginals — before adaptation{map_err_str}",
            fontsize=10,
        )
        self.save_fig(plt.gcf(), "pretrain", "pretrained_fwd_new_marginals.png")

        # Backward direction (BidirectionalStrategy only)
        bwd_model = getattr(strategy, "initial_backward_model", None)
        if bwd_model is None or not isinstance(strategy, BidirectionalStrategy):
            return

        inv_map = getattr(self, "_inverse_transport_map", None)
        y_new = trainer.new_target.sample(n).to(device)
        bwd_model.eval()
        with torch.no_grad():
            evolved_bwd = trainer.flow_matching.evolve(
                y_new, bwd_model, num_steps=100, sigma=0.0, extend=True
            )
            x_hat = evolved_bwd[:, :dim]
        x_true = trainer.new_source.sample(n).to(device)

        map_err_bwd_str = ""
        if inv_map is not None:
            with torch.no_grad():
                err_bwd = (x_hat - inv_map(y_new)).pow(2).sum(-1).mean().sqrt()
            map_err_bwd_str = f"  (map err = {err_bwd.item() * s:.4f})"

        self._display_evolution(
            (y_new * s).cpu(), (x_hat * s).cpu(), (x_true * s).cpu(),
        )
        plt.suptitle(
            f"Pretrained backward flow on new marginals — before adaptation{map_err_bwd_str}",
            fontsize=10,
        )
        self.save_fig(plt.gcf(), "pretrain", "pretrained_bwd_new_marginals.png")

    # ------------------------------------------------------------------
    # Preprocess lifecycle hooks (no-ops)
    # ------------------------------------------------------------------

    def at_preprocess_start(self, iteration: int, trainer: "Trainer", ctx: dict) -> None:
        pass

    def during_preprocess(self, iteration: int, trainer: "Trainer", ctx: dict) -> None:
        pass

    def at_preprocess_end(self, iteration: int, trainer: "Trainer", ctx: dict) -> None:
        pass

    # ------------------------------------------------------------------
    # Duals lifecycle hooks
    # ------------------------------------------------------------------

    def at_duals_start(self, iteration: int, trainer: "Trainer", ctx: dict) -> None:
        pass

    def during_duals(self, epoch: int, iteration: int, trainer: "Trainer", ctx: dict) -> None:
        pass

    def at_duals_end(self, iteration: int, trainer: "Trainer", ctx: dict) -> None:
        pass

    # ------------------------------------------------------------------
    # Flow matching lifecycle hooks
    # ------------------------------------------------------------------

    def at_flow_matching_start(self, iteration: int, trainer: "Trainer", ctx: dict) -> dict:
        """Return a start_ctx dict passed unchanged to at_flow_matching_end."""
        return {}

    def during_flow_matching(self, epoch: int, iteration: int, trainer: "Trainer", ctx: dict) -> None:
        pass

    def at_flow_matching_end(self, iteration: int, trainer: "Trainer", ctx: dict, start_ctx: dict) -> None:
        pass

    # ------------------------------------------------------------------
    # Postprocess lifecycle hooks (no-ops; overridden in logging subclasses)
    # ------------------------------------------------------------------

    def at_postprocess_start(self, iteration: int, trainer: "Trainer", ctx: dict) -> None:
        pass

    def during_postprocess(self, iteration: int, trainer: "Trainer", ctx: dict) -> None:
        pass

    def at_postprocess_end(self, iteration: int, trainer: "Trainer", ctx: dict) -> None:
        pass

    # ------------------------------------------------------------------
    # Diagnostic: sliced Wasserstein distances between marginals
    # ------------------------------------------------------------------

    def _sliced_wasserstein(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        n_proj: int = 200,
        p: int = 1,
    ) -> float:
        d = X.shape[1]
        device = X.device
        dtype = X.dtype
        dirs = torch.randn(d, n_proj, device=device, dtype=dtype)
        dirs = dirs / dirs.norm(dim=0, keepdim=True).clamp(min=1e-8)
        X_proj = X @ dirs
        Y_proj = Y @ dirs
        X_sorted = X_proj.sort(dim=0).values
        Y_sorted = Y_proj.sort(dim=0).values
        n, m = X_sorted.shape[0], Y_sorted.shape[0]
        if n != m:
            n_q = min(n, m)
            idx_X = torch.linspace(0, n - 1, n_q, device=device).long()
            idx_Y = torch.linspace(0, m - 1, n_q, device=device).long()
            X_sorted = X_sorted[idx_X]
            Y_sorted = Y_sorted[idx_Y]
        diff = (X_sorted - Y_sorted).abs()
        if p == 1:
            return diff.mean().item()
        else:
            return diff.pow(2).mean().sqrt().item()

    def analyse(
        self,
        time_step: int,
        trainer,
        n_samples: int = 2000,
        n_proj: int = 2000,
    ) -> dict:
        device = trainer.device
        config = trainer.config
        t = time_step / config.iterations
        t_tensor = torch.full((n_samples, 1), t, device=device)

        with torch.no_grad():
            mu_t = trainer.flow_matching.interpolate(
                trainer.initial_coupling.sample_init(n_samples).to(device),
                trainer.new_source.sample(n_samples).to(device),
                t_tensor, sigma=0.0,
            )
            nu_t = trainer.flow_matching.interpolate(
                trainer.initial_coupling.sample_target(n_samples).to(device),
                trainer.new_target.sample(n_samples).to(device),
                t_tensor, sigma=0.0,
            )
            x_coupling, y_coupling = trainer.current_coupling.coupling_sample(n_samples)
            x_coupling = x_coupling.to(device)
            y_coupling = y_coupling.to(device)

        # Rescale to original data space so metrics are comparable across runs
        # with different scale_factor values (all internal tensors are / scale).
        s = getattr(trainer, "scale", 1.0)
        n_x = x_coupling.shape[0]
        n_y = y_coupling.shape[0]
        mu_t_flat = (mu_t * s).view(n_samples, -1).float()
        nu_t_flat = (nu_t * s).view(n_samples, -1).float()
        x_flat = (x_coupling * s).view(n_x, -1).float()
        y_flat = (y_coupling * s).view(n_y, -1).float()

        return {
            "W1_mu": self._sliced_wasserstein(mu_t_flat, x_flat, n_proj=n_proj, p=1),
            "W2_mu": self._sliced_wasserstein(mu_t_flat, x_flat, n_proj=n_proj, p=2),
            "W1_nu": self._sliced_wasserstein(nu_t_flat, y_flat, n_proj=n_proj, p=1),
            "W2_nu": self._sliced_wasserstein(nu_t_flat, y_flat, n_proj=n_proj, p=2),
        }

    def transport_cost(self, x: torch.Tensor, y: torch.Tensor):
        if self.transport_map is None:
            return None
        return y - self.transport_map(x)

    def _display_evolution(
        self,
        start_points: torch.Tensor,
        end_points: torch.Tensor,
        target_points: Optional[torch.Tensor] = None,
        trajectories: Optional[torch.Tensor] = None,
        trajectory_stride: int = 1,
    ):
        fig, ax = plt.subplots(figsize=(8, 8))
        n_traj = min(self.config.n_trajectories, start_points.shape[0])
        segments = []

        # For dim > 2, project onto first two principal components for visualisation
        dim = start_points.shape[1] if start_points.ndim == 2 else 2
        if dim > 2:
            all_pts = torch.cat([start_points, end_points], dim=0).numpy()
            mean = all_pts.mean(axis=0, keepdims=True)
            centered = all_pts - mean
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)
            proj = Vt[:2]  # (2, dim)
            def _project(t: torch.Tensor) -> np.ndarray:
                return (t.numpy() - mean) @ proj.T
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax_label = " (PCA projection)"
        else:
            def _project(t: torch.Tensor) -> np.ndarray:
                return t.numpy()[:, :2]
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax_label = ""

        start_2d = _project(start_points)
        end_2d   = _project(end_points)

        if trajectories is not None:
            traj_pts = trajectories[::max(1, trajectory_stride), :n_traj, :]  # (T, n_traj, dim)
            T_steps, _, _ = traj_pts.shape
            traj_2d = np.stack([_project(traj_pts[t]) for t in range(T_steps)], axis=0)  # (T, n_traj, 2)
            if T_steps == 0 or not np.array_equal(traj_2d[-1], end_2d[:n_traj]):
                traj_2d = np.concatenate([traj_2d, end_2d[:n_traj][None]], axis=0)
            for j in range(n_traj):
                pts = traj_2d[:, j, :]
                segs = np.stack([pts[:-1], pts[1:]], axis=1)
                segments.extend(segs)
            mid_points_2d = traj_2d[len(traj_2d) // 2]
        else:
            for j in range(n_traj):
                seg = np.stack([start_2d[j], end_2d[j]], axis=0)  # (2, 2)
                segments.append(seg)
            mid_points_2d = (start_2d + end_2d) / 2

        lc = LineCollection(segments, colors="black", alpha=0.2, linewidths=0.8)
        ax.add_collection(lc)

        target_2d = _project(target_points) if target_points is not None else None
        if target_2d is not None:
            ax.scatter(target_2d[:, 0], target_2d[:, 1], c="orange", s=20, alpha=0.5, zorder=4, label="Target", edgecolors='k')
        # ax.scatter(mid_points_2d[:n_traj, 0], mid_points_2d[:n_traj, 1], c="blue", s=20, alpha=0.5, zorder=4, label="Mid", edgecolors='k')
        ax.scatter(start_2d[:n_traj, 0], start_2d[:n_traj, 1], c="red", s=20, alpha=0.5, zorder=4, label="Start", edgecolors='k')
        ax.scatter(end_2d[:n_traj, 0], end_2d[:n_traj, 1], c="green", s=20, alpha=0.5, zorder=4, label="End", edgecolors='k')
        # ax.xaxis.set_major_locator(MultipleLocator(0.25))
        # ax.yaxis.set_major_locator(MultipleLocator(0.25))
        ax.grid(True, alpha=0.2)
        ax.tick_params(axis='both', which='major', labelsize=5)
        ax.legend()
        ax.set_aspect("equal")


class Visualizer3D(Visualizer):
    def __init__(self, config: VisualizeConfig, transport_map: Callable = None):
        super().__init__(config, transport_map)
        self.dim = 3

    def _display_evolution(
        self,
        start_points: torch.Tensor,
        end_points: torch.Tensor,
        target_points: Optional[torch.Tensor] = None,
        trajectories: Optional[torch.Tensor] = None,
        trajectory_stride: int = 1,
    ):
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        from matplotlib.colors import Normalize
        import matplotlib.cm as cm

        # --- publication palette ---
        C_SRC  = "#2166AC"   # blue   – source μ
        C_MID  = "#762A83"   # purple – mid-point
        C_END  = "#D6604D"   # red    – transported end
        C_TGT  = "#4DAC26"   # green  – reference target ν
        TRAJ_CMAP = "plasma"

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="3d")
        fig.patch.set_facecolor("#FAFAFA")
        ax.set_facecolor("#FAFAFA")
        # transparent panes with subtle edges
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("#CCCCCC")

        n_traj = min(self.config.n_trajectories, start_points.shape[0])

        if trajectories is not None:
            traj_np = trajectories[::max(1, trajectory_stride), :n_traj, :3].numpy()
            T = traj_np.shape[0]
            norm = Normalize(vmin=0, vmax=max(T - 2, 1))
            for j in range(n_traj):
                pts = traj_np[:, j, :]          # (T, 3)
                segs = np.stack([pts[:-1], pts[1:]], axis=1)   # (T-1, 2, 3)
                lc = Line3DCollection(segs, cmap=TRAJ_CMAP, norm=norm,
                                      linewidth=0.9, alpha=0.45, zorder=5)
                lc.set_array(np.arange(T - 1).astype(float))
                ax.add_collection(lc)  # type: ignore
            mid_points = torch.from_numpy(traj_np[T // 2])
        else:
            T, norm = None, None
            for j in range(n_traj):
                pts = torch.stack([start_points[j], end_points[j]], dim=0).numpy()
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                        c="#999999", alpha=0.30, linewidth=0.7, zorder=5)
            mid_points = (start_points[:n_traj] + end_points[:n_traj]) / 2

        # data-adaptive limits
        all_pts = torch.cat([start_points[:n_traj], end_points[:n_traj]], dim=0)
        if target_points is not None:
            all_pts = torch.cat([all_pts, target_points], dim=0)
        lo = all_pts.min().item(); hi = all_pts.max().item()
        pad = (hi - lo) * 0.12 + 0.15
        L_lo, L_hi = lo - pad, hi + pad

        kw_s: dict = dict(depthshade=True, edgecolors="none")
        if target_points is not None:
            ax.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2],
                       c=C_TGT, s=8, alpha=0.30, label=r"Target $\nu$", zorder=2, **kw_s)
        ax.scatter(start_points[:n_traj, 0], start_points[:n_traj, 1], start_points[:n_traj, 2],
                   c=C_SRC, s=14, alpha=0.75, label=r"Source $\mu$", zorder=6, **kw_s)
        ax.scatter(end_points[:n_traj, 0], end_points[:n_traj, 1], end_points[:n_traj, 2],
                   c=C_END, s=14, alpha=0.75, label="Transported", zorder=6, **kw_s)
        # ax.scatter(mid_points[:, 0], mid_points[:, 1], mid_points[:, 2],
        #            c=C_MID, s=8, alpha=0.55, label="Mid-point", zorder=5, **kw_s)

        ax.set_xlim([L_lo, L_hi]); ax.set_ylim([L_lo, L_hi]); ax.set_zlim([L_lo, L_hi])
        ax.set_xlabel(r"$x_1$", labelpad=3, fontsize=9)
        ax.set_ylabel(r"$x_2$", labelpad=3, fontsize=9)
        ax.set_zlabel(r"$x_3$", labelpad=3, fontsize=9)  # type: ignore
        ax.tick_params(labelsize=7, pad=1)
        ax.grid(True, alpha=0.15, linewidth=0.5)
        ax.view_init(elev=self.config.camera_elev, azim=self.config.camera_azim)  # type: ignore
        ax.set_box_aspect([1, 1, 1])  # type: ignore

        leg = ax.legend(fontsize=8, loc="upper left", framealpha=0.88,
                        edgecolor="#CCCCCC", markerscale=1.6)
        leg.get_frame().set_linewidth(0.5)

        if trajectories is not None and T is not None and norm is not None:
            sm = cm.ScalarMappable(cmap=TRAJ_CMAP, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, shrink=0.42, pad=0.10, aspect=22)
            cbar.set_label(r"Flow time $t$", fontsize=8)
            cbar.ax.tick_params(labelsize=7)


class VisualizerImage(Visualizer):
    def __init__(self, config: VisualizeConfig, img_shape: Tuple[int, int, int], transport_map: Callable = None):
        super().__init__(config, transport_map)
        self.img_shape = img_shape  # (C, H, W)
        self.dim = img_shape[0] * img_shape[1] * img_shape[2]

    def display(self, epoch: int, trainer: "Trainer"):
        if epoch % self.config.show_each == 0:
            n_samples = min(self.config.n_trajectories, 6)
            x, y_target = trainer.current_coupling.coupling_sample(n_samples)
            x, y_target = x.to(trainer.device), y_target.to(trainer.device)
            y = trainer.flow_matching.evolve(
                x, trainer.strategy.get_flow(epoch, trainer.device)[0],
                num_steps=25, sigma=trainer._sigma, extend=True,
            )
            if is_image_tensor(y):
                y = y[:, :self.img_shape[0], :, :]
            s = trainer.scale
            self._display_evolution((x * s).cpu(), (y * s).cpu(), (y_target * s).cpu())
            plt.show()

    def during_pretrain(self, epoch: int, trainer: "Trainer", ctx: dict) -> None:
        """Image-specific pretrain hook: uses evolve (not trajectory) and limits to 6 samples."""
        phase = ctx.get("phase", "bridges")
        if phase == "bridges":
            if self.config.show_each_marginal <= 0:
                return
            force = ctx.get("force", False)
            if not force and (epoch + 1) % self.config.show_each_marginal != 0:
                return
            mu_model = ctx["mu_model"]
            nu_model = ctx["nu_model"]
            phase_idx = ctx.get("phase_idx", 0)
            n = min(self.config.n_trajectories, 6)
            s = trainer.scale
            for model, source_dist, target_dist, label in [
                (mu_model, trainer.mu_distribution, trainer.new_source, "Mu"),
                (nu_model, trainer.nu_distribution, trainer.new_target, "Nu"),
            ]:
                start = source_dist.sample(n).to(trainer.device)
                with torch.no_grad():
                    end = trainer.flow_matching.evolve(
                        start, model, num_steps=50, sigma=trainer._sigma, extend=False
                    )
                target = target_dist.sample(n).to(trainer.device)
                self._display_evolution((start * s).cpu(), (end * s).cpu(), (target * s).cpu())
                plt.suptitle(f"{label} Marginal Flow — Epoch {epoch} Phase {phase_idx}")
                self.save_fig(
                    plt.gcf(), "pretrain",
                    f"{label.lower()}_bridge_{epoch:05d}_p{phase_idx}.png",
                )

        elif phase == "flow":
            if self.config.show_each_pretrain <= 0:
                return
            if epoch % self.config.show_each_pretrain != 0:
                return
            model = ctx["model"]
            backward = ctx.get("backward", False)
            n = min(self.config.n_trajectories, 6)
            s = trainer.scale
            x0, x1 = trainer.initial_coupling.coupling_sample(n)
            x0, x1 = x0.to(trainer.device), x1.to(trainer.device)
            if backward:
                x0, x1 = x1, x0
            with torch.no_grad():
                end = trainer.flow_matching.evolve(
                    x0, model, num_steps=50, sigma=trainer._sigma, extend=True
                )
                if is_image_tensor(end):
                    end = end[:, :self.img_shape[0], :, :]
            direction = "Backward" if backward else "Forward"
            self._display_evolution((x0 * s).cpu(), (end * s).cpu(), (x1 * s).cpu())
            plt.suptitle(f"Pretrain {direction} Flow — Epoch {epoch}")
            self.save_fig(
                plt.gcf(), "pretrain",
                f"pretrain_{'bwd' if backward else 'fwd'}_{epoch:04d}.png",
            )

    def at_pretrain_end(self, trainer: "Trainer", ctx: dict) -> None:
        """Image-specific: uses evolve and 6 samples."""
        n = min(self.config.n_trajectories, 6)
        s = trainer.scale
        for model_attr, src_attr, tgt_attr, title, fname in [
            ("mu_bridge", "mu_distribution", "new_source",
             "Mu bridge: source distribution flow", "mu_bridge_final.png"),
            ("nu_bridge", "nu_distribution", "new_target",
             "Nu bridge: target distribution flow", "nu_bridge_final.png"),
        ]:
            model = getattr(trainer.strategy, model_attr)
            start = getattr(trainer, src_attr).sample(n).to(trainer.device)
            model.eval()
            with torch.no_grad():
                end = trainer.flow_matching.evolve(
                    start, model, num_steps=50, sigma=trainer._sigma, extend=False
                )
            target = getattr(trainer, tgt_attr).sample(n).to(trainer.device)
            self._display_evolution((start * s).cpu(), (end * s).cpu(), (target * s).cpu())
            plt.suptitle(title)
            self.save_fig(plt.gcf(), "marginals", fname)

    def at_flow_pretrain_end(self, trainer: "Trainer", ctx: dict) -> None:
        """Image version: 3-row grid (source | pretrained output | true target).

        Shows the pretrained forward (and backward) flows applied to the new
        marginals before any dual adaptation.  Each column is one image sample;
        rows are: source, transported by pretrained model, true target.
        Saves to ``pretrain/pretrained_fwd_new_marginals.png``.
        """
        strategy = trainer.strategy
        fwd_model = getattr(strategy, "initial_forward_model", None)
        if fwd_model is None:
            return

        n = min(self.config.n_trajectories, 6)
        device = trainer.device
        s = trainer.scale
        C = self.img_shape[0]

        x_new = trainer.new_source.sample(n).to(device)
        fwd_model.eval()
        with torch.no_grad():
            evolved = trainer.flow_matching.evolve(
                x_new, fwd_model, num_steps=100, sigma=0.0, extend=True
            )
            y_hat = evolved[:, :C, :, :] if is_image_tensor(evolved) else evolved[:, :C]
        y_true = trainer.new_target.sample(n).to(device)

        self._display_evolution(
            (x_new * s).cpu(), (y_hat * s).cpu(), (y_true * s).cpu()
        )
        plt.suptitle(
            "Pretrained forward flow on new marginals — before adaptation",
            fontsize=10,
        )
        self.save_fig(plt.gcf(), "pretrain", "pretrained_fwd_new_marginals.png")

        # Backward direction (BidirectionalStrategy only)
        bwd_model = getattr(strategy, "initial_backward_model", None)
        if bwd_model is None or not isinstance(strategy, BidirectionalStrategy):
            return

        y_new = trainer.new_target.sample(n).to(device)
        bwd_model.eval()
        with torch.no_grad():
            evolved_bwd = trainer.flow_matching.evolve(
                y_new, bwd_model, num_steps=100, sigma=0.0, extend=True
            )
            x_hat = evolved_bwd[:, :C, :, :] if is_image_tensor(evolved_bwd) else evolved_bwd[:, :C]
        x_true = trainer.new_source.sample(n).to(device)

        self._display_evolution(
            (y_new * s).cpu(), (x_hat * s).cpu(), (x_true * s).cpu()
        )
        plt.suptitle(
            "Pretrained backward flow on new marginals — before adaptation",
            fontsize=10,
        )
        self.save_fig(plt.gcf(), "pretrain", "pretrained_bwd_new_marginals.png")

    def _display_evolution(
        self, start_points: torch.Tensor, end_points: torch.Tensor,
        target_points: Optional[torch.Tensor] = None, **kwargs,
    ):
        n = min(start_points.shape[0], 6)
        rows = 3 if target_points is not None else 2
        fig, axes = plt.subplots(rows, n, figsize=(2 * n, 2 * rows))
        if n == 1:
            axes = axes.reshape(rows, 1)
        C, H, W = self.img_shape

        def _show(ax, tensor, i, title):
            img = tensor[i]  # (C, H, W)
            if img.dim() == 3 and img.shape[0] == 1:
                img = img.squeeze(0)          # (H, W) — grayscale
                cmap = "gray"
            elif img.dim() == 3:
                img = img.permute(1, 2, 0)   # (H, W, C) — RGB
                img = img.clamp(-1, 1)
                cmap = None
            else:
                img = img.reshape(H, W)
                cmap = "gray"
            ax.imshow(img.float().numpy(), cmap=cmap, vmin=-1, vmax=1)
            ax.set_title(title, fontsize=7); ax.axis("off")

        for i in range(n):
            _show(axes[0, i], start_points, i, "Source")
            _show(axes[1, i], end_points, i, "Transported")
            if target_points is not None:
                _show(axes[2, i], target_points, i, "Target")
        plt.tight_layout()


def plot_2d_flow_trajectories(
    trajectories: torch.Tensor,
    t: torch.Tensor,
    n_trajectories: int = 50,
    target_points: Optional[torch.Tensor] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    title: str = "Flow Trajectories",
):
    """Plot sample flow trajectories."""
    fig, ax = plt.subplots(figsize=(8, 8))
    trajectories_np = trajectories.cpu().numpy()
    n_trajectories = min(n_trajectories, trajectories.shape[1])

    segments = []
    for j in range(n_trajectories):
        points = trajectories_np[:, j, :2]
        segs = np.stack([points[:-1], points[1:]], axis=1)
        segments.extend(segs)

    start_points = trajectories_np[0, :, :2]
    end_points = trajectories_np[-1, :, :2]

    if target_points is not None:
        ax.scatter(target_points[:, 0], target_points[:, 1], c="orange", s=5, alpha=0.5, zorder=4, label="Target")
    ax.scatter(start_points[:, 0], start_points[:, 1], c="red", s=5, alpha=0.5, zorder=4, label="Start")
    ax.scatter(end_points[:, 0], end_points[:, 1], c="green", s=5, alpha=0.5, zorder=4, label="End")
    mid_points = trajectories_np[trajectories_np.shape[0] // 2, :, :2]
    # ax.scatter(mid_points[:, 0], mid_points[:, 1], c="blue", s=5, alpha=0.5, zorder=4, label="Mid")
    ax.grid(True, alpha=0.2)
    ax.set_title(title)
    lc = LineCollection(segments, colors="gray", alpha=0.5, linewidths=0.8, zorder=1)
    ax.add_collection(lc)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.legend()
