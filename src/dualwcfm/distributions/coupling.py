from abc import ABC, abstractmethod
import math
import os
import tempfile
import shutil

from typing import Callable, Optional, Tuple, Union
import numpy as np
import torch

from dualwcfm.distributions.distribution import Distribution


class MarginalDistribution(Distribution):
    def __init__(self, sample_func):
        self.sample_func = sample_func

    def sample(self, n: int) -> torch.Tensor:
        return self.sample_func(n)

    def clone(self):
        import copy

        return copy.deepcopy(self)


class CouplingDistribution(Distribution, ABC):
    @abstractmethod
    def coupling_sample(self, n: int) -> Union[tuple[torch.Tensor, torch.Tensor], list[torch.Tensor], torch.Tensor]:
        pass

    @abstractmethod
    def sample_init(self, n: int) -> torch.Tensor:
        pass

    @abstractmethod
    def sample_target(self, n: int) -> torch.Tensor:
        pass

    def get_marginals(self) -> tuple[Distribution, Distribution]:
        return MarginalDistribution(self.sample_init), MarginalDistribution(self.sample_target)

    def sample(self, n: int) -> torch.Tensor:
        return super().coupling_sample(n)  # type: ignore

    def clone(self):
        # Clone the coupling distribution and everything inside
        import copy

        return copy.deepcopy(self)


class FunctionalCouplingDistribution(CouplingDistribution):
    def __init__(self, coupling_func):
        self.coupling_func = coupling_func

    def sample(self, n: int) -> torch.Tensor:
        x0, x1 = self.coupling_func(n)
        return torch.stack([x0, x1], dim=0)

    def sample_init(self, n: int) -> torch.Tensor:
        x0, _ = self.coupling_func(n)
        return x0

    def sample_target(self, n: int) -> torch.Tensor:
        _, x1 = self.coupling_func(n)
        return x1

    def coupling_sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.coupling_func(n)


class RotatedShapeCoupling(CouplingDistribution):
    def __init__(
        self,
        centers,
        source_scale=0.1,
        target_scale=0.1,
        rotation_degrees=45.0,
        source_shift=(0, 0),
        target_shift=(0, 0),
        transformation: Callable[[torch.Tensor], torch.Tensor] = None,  # New parameter
        device=None,
    ):
        super().__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.centers = torch.tensor(centers, dtype=torch.float32).to(self.device)
        self.source_scale = source_scale
        self.target_scale = target_scale
        self.source_shift = torch.tensor(source_shift, dtype=torch.float32).to(self.device)
        self.target_shift = torch.tensor(target_shift, dtype=torch.float32).to(self.device)

        # Set transformation to identity if none is provided
        self.transformation = transformation if transformation is not None else lambda x: x

        theta = torch.tensor(rotation_degrees * math.pi / 180.0)
        c, s = torch.cos(theta), torch.sin(theta)
        self.R = torch.tensor([[c, -s], [s, c]]).to(self.device)

        self.n_components = self.centers.shape[0]
        self.cat_dist = torch.distributions.Categorical(torch.ones(self.n_components, device=self.device))

    def sample_init(self, n: int) -> torch.Tensor:
        x, _ = self.coupling_sample(n)
        return x

    def sample_target(self, n: int) -> torch.Tensor:
        _, y = self.coupling_sample(n)
        return y

    def coupling_sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        indices = self.cat_dist.sample((torch.Size([n])))
        chosen_centers = self.centers[indices]

        noise_x = torch.randn(n, 2, device=self.device)
        x_centered = chosen_centers + noise_x * self.source_scale

        # 1. Final source coordinates
        x = x_centered + self.source_shift

        # 2. Apply transformation T(x) before rotation
        x_transformed = self.transformation(x_centered)

        # 3. Rotate the transformed coordinates and scale
        y_centered = (x_transformed @ self.R.T) * (self.target_scale / self.source_scale)
        y = y_centered + self.target_shift

        return x, y

    def transform_source(self, x: torch.Tensor) -> torch.Tensor:
        x_centered = x - self.source_shift
        x_transformed = self.transformation(x_centered)
        y_centered = (x_transformed @ self.R.T) * (self.target_scale / self.source_scale)
        y = y_centered + self.target_shift
        return y


class FixedSizeCouplingDistribution(CouplingDistribution):

    def __init__(self, coupling_distribution: CouplingDistribution, size: int = 1000):
        self.coupling_distribution = coupling_distribution
        self.size = size
        self.dataset = torch.stack(list(self.coupling_distribution.coupling_sample(self.size)), dim=0)

    @staticmethod
    def from_dataset(dataset: Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]):
        obj = FixedSizeCouplingDistribution.__new__(FixedSizeCouplingDistribution)
        obj.size = dataset[0].shape[0]
        obj.dataset = dataset
        return obj

    def clone(self):
        import copy
        return copy.deepcopy(self)

    def coupling_sample(self, n: int):
        idx = torch.randint(0, self.size, (n,))
        return (self.dataset[0][idx], self.dataset[1][idx])

    def sample_init(self, n: int):
        idx = torch.randint(0, self.size, (n,))
        return self.dataset[0][idx]

    def sample_target(self, n: int):
        idx = torch.randint(0, self.size, (n,))
        return self.dataset[1][idx]

    def update_chunk(self, start_idx: int, x_chunk: torch.Tensor, y_chunk: torch.Tensor):
        """Updates a chunk of the dataset in-place."""
        end_idx = start_idx + x_chunk.shape[0]

        self.dataset[0][start_idx:end_idx] = x_chunk
        self.dataset[1][start_idx:end_idx] = y_chunk

class DiskCouplingDistribution(CouplingDistribution):
    """Disk-based coupling distribution using numpy memmap for fast I/O.

    Stores all samples in two flat binary files (x.dat, y.dat) backed by
    numpy memmap. Writes are raw memory copies with no serialization overhead,
    making update_chunk ~10-100x faster than the previous chunk-based pickle
    approach. A small meta.pt records shape/dtype so the files can be reopened
    across processes or after a crash.
    """

    def __init__(
        self,
        coupling_distribution: Optional[CouplingDistribution] = None,
        size: int = 1000,
        save_dir: Optional[str] = None,
        chunk_size: int = 256,
        device: torch.device = torch.device("cpu"),
    ):
        self.size = size
        self.chunk_size = chunk_size  # kept for step_postprocess compatibility
        self.device = device
        self._is_temp = save_dir is None
        self._x_mmap: Optional[np.ndarray] = None
        self._y_mmap: Optional[np.ndarray] = None

        if save_dir is None:
            self.save_dir = tempfile.mkdtemp(prefix="coupling_disk_")
        else:
            self.save_dir = save_dir
            os.makedirs(save_dir, exist_ok=True)

        if coupling_distribution is not None:
            self._initialize_from_distribution(coupling_distribution)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _x_path(self) -> str:
        return os.path.join(self.save_dir, "x.dat")

    def _y_path(self) -> str:
        return os.path.join(self.save_dir, "y.dat")

    def _meta_path(self) -> str:
        return os.path.join(self.save_dir, "meta.pt")

    def _init_memmap(self, sample_shape: tuple, dtype: np.dtype):
        """Create the two memmap files with the given per-sample shape."""
        meta = {"sample_shape": list(sample_shape), "dtype": str(dtype), "size": self.size}
        torch.save(meta, self._meta_path())
        full_shape = (self.size, *sample_shape)
        self._x_mmap = np.memmap(self._x_path(), dtype=dtype, mode="w+", shape=full_shape)
        self._y_mmap = np.memmap(self._y_path(), dtype=dtype, mode="w+", shape=full_shape)

    def _ensure_memmap(self):
        """Re-open existing memmap files if not already loaded."""
        if self._x_mmap is not None:
            return
        if not os.path.exists(self._meta_path()):
            return
        meta = torch.load(self._meta_path(), weights_only=True)
        sample_shape = tuple(meta["sample_shape"])
        dtype = np.dtype(meta["dtype"])
        full_shape = (self.size, *sample_shape)
        self._x_mmap = np.memmap(self._x_path(), dtype=dtype, mode="r+", shape=full_shape)
        self._y_mmap = np.memmap(self._y_path(), dtype=dtype, mode="r+", shape=full_shape)

    def _initialize_from_distribution(self, coupling_distribution: CouplingDistribution):
        """Sample from distribution and write to memmap in chunks."""
        n_chunks = (self.size + self.chunk_size - 1) // self.chunk_size
        print(f"Initializing disk coupling at {self.save_dir} ({self.size} samples)...")

        for chunk_idx in range(n_chunks):
            start = chunk_idx * self.chunk_size
            end = min(start + self.chunk_size, self.size)

            x, y = coupling_distribution.coupling_sample(end - start)
            x_np = x.cpu().numpy()
            y_np = y.cpu().numpy()

            if self._x_mmap is None:
                self._init_memmap(x_np.shape[1:], x_np.dtype)

            self._x_mmap[start:end] = x_np
            self._y_mmap[start:end] = y_np

        if self._x_mmap is not None:
            self._x_mmap.flush()
            self._y_mmap.flush()

        print(f"Disk coupling initialized: {self.size} samples")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def n_chunks(self) -> int:
        return (self.size + self.chunk_size - 1) // self.chunk_size

    def _load_chunk(self, chunk_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read a chunk by index from the memmap (compatibility shim for callers)."""
        self._ensure_memmap()
        start = chunk_idx * self.chunk_size
        end = min(start + self.chunk_size, self.size)
        x = torch.from_numpy(self._x_mmap[start:end].copy())
        y = torch.from_numpy(self._y_mmap[start:end].copy())
        return x, y

    def update_chunk(self, start_idx: int, x_chunk: torch.Tensor, y_chunk: torch.Tensor):
        """Update a contiguous range via direct memmap slice write (no serialization)."""
        x_np = x_chunk.cpu().numpy()
        y_np = y_chunk.cpu().numpy()
        end_idx = start_idx + x_np.shape[0]

        if self._x_mmap is None:
            # First call: deduce shape and create the memmap files
            self._init_memmap(x_np.shape[1:], x_np.dtype)

        self._x_mmap[start_idx:end_idx] = x_np
        self._y_mmap[start_idx:end_idx] = y_np
        # No explicit flush — the OS write-back is sufficient for same-process reads.
        # Call flush() explicitly only before cross-process access (e.g. clone).

    def coupling_sample(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample n random pairs via numpy fancy indexing (returns copies)."""
        self._ensure_memmap()
        indices = torch.randint(0, self.size, (n,)).numpy()
        x = torch.from_numpy(self._x_mmap[indices].copy())
        y = torch.from_numpy(self._y_mmap[indices].copy())
        return x.to(self.device), y.to(self.device)

    def sample_init(self, n: int) -> torch.Tensor:
        x, _ = self.coupling_sample(n)
        return x

    def sample_target(self, n: int) -> torch.Tensor:
        _, y = self.coupling_sample(n)
        return y

    def clone(self):
        """Clone by copying x.dat, y.dat, meta.pt to a new temp directory."""
        new_save_dir = tempfile.mkdtemp(prefix="coupling_disk_clone_")

        # Flush pending writes before copying
        if self._x_mmap is not None:
            self._x_mmap.flush()
            self._y_mmap.flush()

        for fname in ("x.dat", "y.dat", "meta.pt"):
            src = os.path.join(self.save_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(new_save_dir, fname))

        cloned = DiskCouplingDistribution.__new__(DiskCouplingDistribution)
        cloned.size = self.size
        cloned.chunk_size = self.chunk_size
        cloned.device = self.device
        cloned.save_dir = new_save_dir
        cloned._is_temp = True
        cloned._x_mmap = None
        cloned._y_mmap = None
        cloned._ensure_memmap()
        return cloned

    def cleanup(self):
        """Close memmaps and remove the directory."""
        if self._is_temp and os.path.exists(self.save_dir):
            # Must delete memmap objects before removing files on Windows
            self._x_mmap = None
            self._y_mmap = None
            shutil.rmtree(self.save_dir)

    def __del__(self):
        if hasattr(self, "_is_temp") and self._is_temp:
            try:
                self.cleanup()
            except Exception:
                pass

    @staticmethod
    def from_directory(save_dir: str, size: int, chunk_size: int = 256, device: torch.device = torch.device("cpu")):
        """Load an existing disk coupling from a directory containing x.dat/y.dat/meta.pt."""
        dist = DiskCouplingDistribution.__new__(DiskCouplingDistribution)
        dist.size = size
        dist.chunk_size = chunk_size
        dist.device = device
        dist.save_dir = save_dir
        dist._is_temp = False
        dist._x_mmap = None
        dist._y_mmap = None
        dist._ensure_memmap()
        return dist


class WeightedCouplingDistribution(CouplingDistribution):
    def __init__(self, base_fixed_coupling: CouplingDistribution, size: int = 1000):
        self.base_fixed_coupling = base_fixed_coupling
        self.size = size
        self.dataset = self.base_fixed_coupling.coupling_sample(self.size)
        self.log_weights = torch.zeros(self.size)

    def coupling_sample(self, n: int):  # type: ignore
        idx = torch.randint(0, self.size, (n,))
        samples = (self.dataset[0][idx], self.dataset[1][idx])
        log_w = self.log_weights[idx]
        return samples, log_w

    def sample_init(self, n: int):
        idx = torch.randint(0, self.size, (n,))
        return self.dataset[0][idx]

    def sample_target(self, n: int):
        idx = torch.randint(0, self.size, (n,))
        return self.dataset[1][idx]

    def sample(self, n: int):
        pass  # type: ignore