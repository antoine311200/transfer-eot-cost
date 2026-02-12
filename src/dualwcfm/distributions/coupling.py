from abc import ABC, abstractmethod
from collections import OrderedDict
import math
import os
import tempfile
import shutil

from typing import Callable, Optional, Tuple, Union
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
    """Fixed-size coupling distribution that stores samples for reuse.

    Supports both vector data (N, D) and image data (N, C, H, W).
    """

    def __init__(self, coupling_distribution: CouplingDistribution, size: int = 1000):
        self.coupling_distribution = coupling_distribution
        self.size = size
        x, y = self.coupling_distribution.coupling_sample(self.size)
        # Store as tuple of tensors instead of stacked tensor for better image support
        self.dataset_x = x
        self.dataset_y = y
        # Keep legacy dataset attribute for backward compatibility
        self._update_legacy_dataset()

    def _update_legacy_dataset(self):
        """Update legacy stacked dataset for backward compatibility."""
        # Only stack if both tensors have compatible shapes for stacking
        if self.dataset_x.dim() == self.dataset_y.dim():
            self.dataset = torch.stack([self.dataset_x, self.dataset_y], dim=0)
        else:
            self.dataset = (self.dataset_x, self.dataset_y)

    @staticmethod
    def from_dataset(dataset: Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]):
        obj = FixedSizeCouplingDistribution.__new__(FixedSizeCouplingDistribution)
        if isinstance(dataset, tuple):
            obj.dataset_x = dataset[0]
            obj.dataset_y = dataset[1]
            obj.size = dataset[0].shape[0]
        else:
            obj.dataset_x = dataset[0]
            obj.dataset_y = dataset[1]
            obj.size = dataset[0].shape[0]
        obj._update_legacy_dataset()
        return obj

    def clone(self):
        import copy
        return copy.deepcopy(self)

    def coupling_sample(self, n: int):
        idx = torch.randint(0, self.size, (n,))
        return (self.dataset_x[idx], self.dataset_y[idx])

    def sample_init(self, n: int):
        idx = torch.randint(0, self.size, (n,))
        return self.dataset_x[idx]

    def sample_target(self, n: int):
        idx = torch.randint(0, self.size, (n,))
        return self.dataset_y[idx]

    def update_chunk(self, start_idx: int, x_chunk: torch.Tensor, y_chunk: torch.Tensor):
        """Update a chunk of the dataset (used during training)."""
        end_idx = start_idx + x_chunk.shape[0]
        self.dataset_x[start_idx:end_idx] = x_chunk
        self.dataset_y[start_idx:end_idx] = y_chunk


class DiskCouplingDistribution(CouplingDistribution):
    """Disk-based coupling distribution that stores samples on disk instead of memory.

    Useful for image data where keeping the full dataset in memory would cause OOM.
    Data is stored in chunks on disk and loaded on-demand during sampling.
    """

    def __init__(
        self,
        coupling_distribution: Optional[CouplingDistribution] = None,
        size: int = 1000,
        save_dir: Optional[str] = None,
        chunk_size: int = 256,
        device: torch.device = torch.device("cpu"),
    ):
        """Initialize disk-based coupling distribution.

        Args:
            coupling_distribution: Source distribution to sample initial data from.
                If None, expects data to be written via update_chunk.
            size: Total number of samples to store.
            save_dir: Directory to store data. If None, creates a temp directory.
            chunk_size: Number of samples per chunk file. Smaller = less memory per load.
            device: Device to return samples on (samples are stored on CPU).
        """
        self.size = size
        self.chunk_size = chunk_size
        self.device = device
        self._is_temp = save_dir is None

        # Create save directory
        if save_dir is None:
            self.save_dir = tempfile.mkdtemp(prefix="coupling_disk_")
        else:
            self.save_dir = save_dir
            os.makedirs(save_dir, exist_ok=True)

        self.n_chunks = (size + chunk_size - 1) // chunk_size

        # LRU cache for loaded chunks (avoids re-reading disk on every sample call)
        self._chunk_cache: OrderedDict[int, Tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self._cache_max_size = max(64, (size + chunk_size - 1) // chunk_size)  # Cache all chunks by default

        # Initialize from distribution if provided
        if coupling_distribution is not None:
            self._initialize_from_distribution(coupling_distribution)

    def _chunk_path(self, chunk_idx: int) -> str:
        """Get path for a chunk file."""
        return os.path.join(self.save_dir, f"chunk_{chunk_idx:06d}.pt")

    def _initialize_from_distribution(self, coupling_distribution: CouplingDistribution):
        """Sample from distribution and save to disk in chunks."""
        print(f"Initializing disk coupling at {self.save_dir} with {self.n_chunks} chunks...")

        for chunk_idx in range(self.n_chunks):
            start = chunk_idx * self.chunk_size
            end = min(start + self.chunk_size, self.size)
            n = end - start

            x, y = coupling_distribution.coupling_sample(n)
            self._save_chunk(chunk_idx, x.cpu(), y.cpu())

        print(f"Disk coupling initialized: {self.size} samples in {self.n_chunks} chunks")

    def _save_chunk(self, chunk_idx: int, x: torch.Tensor, y: torch.Tensor):
        """Save a chunk to disk and invalidate its cache entry."""
        torch.save({"x": x, "y": y}, self._chunk_path(chunk_idx))
        self._invalidate_cache(chunk_idx)

    def _load_chunk(self, chunk_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load a chunk from disk with LRU caching."""
        if chunk_idx in self._chunk_cache:
            # Move to end (most recently used)
            self._chunk_cache.move_to_end(chunk_idx)
            return self._chunk_cache[chunk_idx]

        data = torch.load(self._chunk_path(chunk_idx), weights_only=True, map_location="cpu")
        result = (data["x"], data["y"])

        # Add to cache, evict oldest if full
        self._chunk_cache[chunk_idx] = result
        if len(self._chunk_cache) > self._cache_max_size:
            self._chunk_cache.popitem(last=False)

        return result

    def _invalidate_cache(self, chunk_idx: int):
        """Remove a chunk from cache (call after writing to disk)."""
        self._chunk_cache.pop(chunk_idx, None)

    def coupling_sample(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample n random pairs from disk."""
        indices = torch.randint(0, self.size, (n,))
        return self._get_samples_by_indices(indices)

    def _get_samples_by_indices(self, indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load samples for given indices, using batch indexing per chunk."""
        chunk_indices = indices // self.chunk_size
        local_indices = indices % self.chunk_size

        # Pre-allocate output tensors using shape from first chunk
        first_chunk = self._load_chunk(chunk_indices[0].item())
        sample_shape = first_chunk[0].shape[1:]  # Shape without batch dim
        x_out = torch.empty(len(indices), *sample_shape)
        y_out = torch.empty(len(indices), *sample_shape)

        # Batch-index per unique chunk instead of per-sample Python loop
        for chunk_idx in chunk_indices.unique():
            mask = chunk_indices == chunk_idx
            x_chunk, y_chunk = self._load_chunk(chunk_idx.item())
            local = local_indices[mask]
            x_out[mask] = x_chunk[local]
            y_out[mask] = y_chunk[local]

        return x_out.to(self.device), y_out.to(self.device)

    def sample_init(self, n: int) -> torch.Tensor:
        """Sample n source (x) values."""
        x, _ = self.coupling_sample(n)
        return x

    def sample_target(self, n: int) -> torch.Tensor:
        """Sample n target (y) values."""
        _, y = self.coupling_sample(n)
        return y

    def update_chunk(self, start_idx: int, x_chunk: torch.Tensor, y_chunk: torch.Tensor):
        """Update a contiguous range in the dataset.

        This method is called during training to update the coupling after transport.
        It handles the case where the update range spans multiple chunk files.

        Args:
            start_idx: Starting index in the full dataset.
            x_chunk: New x values (source).
            y_chunk: New y values (target).
        """
        n_samples = x_chunk.shape[0]
        end_idx = start_idx + n_samples

        # Determine which chunks are affected
        start_chunk = start_idx // self.chunk_size
        end_chunk = (end_idx - 1) // self.chunk_size

        samples_written = 0

        for chunk_idx in range(start_chunk, end_chunk + 1):
            chunk_start = chunk_idx * self.chunk_size
            chunk_end = min(chunk_start + self.chunk_size, self.size)

            # Intersection of [start_idx, end_idx) and [chunk_start, chunk_end)
            write_start = max(start_idx, chunk_start)
            write_end = min(end_idx, chunk_end)

            if write_start >= write_end:
                continue

            # Indices within the chunk
            local_start = write_start - chunk_start
            local_end = write_end - chunk_start

            # Indices within the input x_chunk/y_chunk
            input_start = write_start - start_idx
            input_end = write_end - start_idx

            # Load existing chunk (or create new if it spans full chunk)
            if local_start == 0 and local_end == (chunk_end - chunk_start):
                # Full chunk replacement - no need to load
                x_existing = x_chunk[input_start:input_end].cpu()
                y_existing = y_chunk[input_start:input_end].cpu()
            else:
                # Partial update - load existing and modify
                x_existing, y_existing = self._load_chunk(chunk_idx)
                x_existing[local_start:local_end] = x_chunk[input_start:input_end].cpu()
                y_existing[local_start:local_end] = y_chunk[input_start:input_end].cpu()

            self._save_chunk(chunk_idx, x_existing, y_existing)
            samples_written += (write_end - write_start)

    def clone(self):
        """Clone the distribution by copying all chunk files to a new directory."""
        import copy

        new_save_dir = tempfile.mkdtemp(prefix="coupling_disk_clone_")

        # Copy all chunk files
        for chunk_idx in range(self.n_chunks):
            src = self._chunk_path(chunk_idx)
            if os.path.exists(src):
                dst = os.path.join(new_save_dir, f"chunk_{chunk_idx:06d}.pt")
                shutil.copy2(src, dst)

        # Create new instance pointing to cloned data
        cloned = DiskCouplingDistribution.__new__(DiskCouplingDistribution)
        cloned.size = self.size
        cloned.chunk_size = self.chunk_size
        cloned.device = self.device
        cloned.save_dir = new_save_dir
        cloned.n_chunks = self.n_chunks
        cloned._is_temp = True  # Cloned temp dirs should be cleaned up
        cloned._chunk_cache = OrderedDict()
        cloned._cache_max_size = self._cache_max_size

        return cloned

    def cleanup(self):
        """Remove the temporary directory and all chunk files."""
        if self._is_temp and os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
            print(f"Cleaned up disk coupling at {self.save_dir}")

    def __del__(self):
        """Cleanup on deletion if using temp directory."""
        # Only cleanup if this is a temp directory
        if hasattr(self, "_is_temp") and self._is_temp:
            try:
                self.cleanup()
            except Exception:
                pass  # Ignore errors during cleanup in destructor

    @staticmethod
    def from_directory(save_dir: str, size: int, chunk_size: int = 256, device: torch.device = torch.device("cpu")):
        """Load an existing disk coupling from a directory.

        Args:
            save_dir: Directory containing chunk files.
            size: Total number of samples.
            chunk_size: Chunk size used when saving.
            device: Device to return samples on.
        """
        dist = DiskCouplingDistribution.__new__(DiskCouplingDistribution)
        dist.size = size
        dist.chunk_size = chunk_size
        dist.device = device
        dist.save_dir = save_dir
        dist.n_chunks = (size + chunk_size - 1) // chunk_size
        dist._is_temp = False
        dist._chunk_cache = OrderedDict()
        dist._cache_max_size = max(64, dist.n_chunks)
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
