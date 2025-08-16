# --- BADGE Sampler -----------------------------------------------------------
# Drop this into active_learning_models.py next to your other samplers.

import math
import os
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset
from active_learning_models import ActiveLearningPipeline

class BADGESampler(ActiveLearningPipeline):  # <-- rename Base class if needed
    """
    Batch Active learning by Diverse Gradient Embeddings (BADGE).

    Key ideas:
      - Uncertainty: gradient magnitude wrt last layer using model's argmax "hallucinated" label.
      - Diversity: k-MEANS++ seeding on those gradient embeddings.

    This implementation:
      - Finds the last linear/classifier layer automatically (ResNet: .fc, DenseNet: .classifier, or last nn.Linear).
      - Uses a forward-hook on that last layer to grab the *input to the last linear* (penultimate features z).
      - Computes gradient embeddings: E = outer(p, z) with row ŷ subtracted by z, then flattens to (K*d).
      - Handles very large pools via chunked inference, optional subsampling, and optional on-disk memmap for distances.

    Expected attributes on the base class (robustly retrieved via getattr with fallbacks):
      - self.model : torch.nn.Module
      - self.device : torch.device or str
      - self.pool_dataset / self.unlabeled_dataset / self.pool_set : torch Dataset or Subset
      - self.pool_indices / self.unlabeled_indices : Optional[List[int]] indices into an underlying dataset
      - self.eval_batch_size / self.batch_size : int for inference
      - self.num_classes : Optional[int] (otherwise inferred from model output)

    Optional BADGE-specific knobs you can set on the instance:
      - self.badge_subsample : Optional[int]   # if set, uniformly subsample this many candidates from the pool
      - self.badge_chunk_size: Optional[int]   # per-inference chunk size (default: eval batch size)
      - self.badge_use_memmap: bool            # store/update k-means++ distance vector on disk (default: False)
      - self.badge_fp16: bool                  # use fp16 for embeddings to reduce memory (default: False)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Defaults (can be overridden on the instance)
        self.badge_subsample: Optional[int] = getattr(self, "badge_subsample", None)
        self.badge_chunk_size: Optional[int] = getattr(self, "badge_chunk_size", None)
        self.badge_use_memmap: bool = getattr(self, "badge_use_memmap", False)
        self.badge_fp16: bool = getattr(self, "badge_fp16", False)

        # Will be populated when we attach a hook on the last linear layer:
        self._penultimate_cache = None
        self._last_linear: Optional[nn.Linear] = None
        self._hook_handle = None

    # ------------------------------------------------------------------ Public (called by base .sample)
    def _sample(self, budget: int) -> List[int]:
        """
        Return a list of *pool indices* (relative to the pool) of size `budget`
        selected by BADGE.
        """
        print(f"[BADGE] Starting BADGE sampling with budget={budget}")
        
        model = getattr(self, "model", None)
        assert model is not None, "BADGESampler expects `self.model`."
        print(f"[BADGE] Model type: {type(model).__name__}")

        device = torch.device(getattr(self, "device", "cpu"))
        print(f"[BADGE] Using device: {device}")
        model.eval().to(device)
        print(f"[BADGE] Model moved to device and set to eval mode")

        # Identify pool dataset and indices
        print(f"[BADGE] Resolving pool dataset and indices...")
        pool_ds, pool_indices = self._resolve_pool()
        n_pool = len(pool_indices)
        print(f"[BADGE] Pool size: {n_pool} samples")

        # Optional subsampling for very large pools
        if self.badge_subsample and self.badge_subsample < n_pool:
            print(f"[BADGE] Subsampling pool from {n_pool} to {self.badge_subsample}")
            seed = getattr(self, "random_seed", None)
            if seed is not None:
                torch.manual_seed(seed)
            pool_indices = torch.randperm(n_pool)[:self.badge_subsample].tolist()
            n_pool = len(pool_indices)
            print(f"[BADGE] After subsampling: {n_pool} samples")

        # Prepare loader
        bs = self._eval_bs()
        chunk_bs = self.badge_chunk_size or bs
        print(f"[BADGE] Creating DataLoader with batch_size={chunk_bs}, pool_size={n_pool}")
        loader = DataLoader(
            Subset(pool_ds, pool_indices),
            batch_size=chunk_bs,
            shuffle=False,
            num_workers=getattr(self, "num_workers", 0),
            pin_memory=True,
        )
        print(f"[BADGE] DataLoader created with {len(loader)} batches")

        # Attach forward hook to capture the *input to* the last linear (penultimate features z)
        print(f"[BADGE] Attaching forward hook to capture penultimate features...")
        self._attach_last_linear_input_hook(model)
        print(f"[BADGE] Forward hook attached successfully")

        # Pass 1: gather penultimate features (z) and logits (for softmax p)
        print(f"[BADGE] Collecting features and probabilities...")
        Z, P = self._collect_features_and_probs(model, loader, device)
        print(f"[BADGE] Features collected: Z.shape={Z.shape}, P.shape={P.shape}")
        self._detach_hook()
        print(f"[BADGE] Forward hook detached")

        # Compute BADGE gradient embeddings (N, K*d)
        print(f"[BADGE] Computing BADGE gradient embeddings...")
        G = self._badge_embeddings(Z, P, fp16=self.badge_fp16)
        print(f"[BADGE] Gradient embeddings computed: G.shape={G.shape}")

        # k-MEANS++ seeding on embeddings to pick `budget` points
        print(f"[BADGE] Running k-means++ seeding to select {budget} points...")
        chosen_local = self._kmeanspp(G, k=budget)
        print(f"[BADGE] k-means++ completed, selected {len(chosen_local)} local indices")

        # Map back to pool indices
        chosen_pool_indices = [pool_indices[i] for i in chosen_local]
        print(f"[BADGE] Mapped to pool indices, returning {len(chosen_pool_indices)} samples")
        return chosen_pool_indices

    # ------------------------------------------------------------------ Feature & prob collection
    @torch.no_grad()
    def _collect_features_and_probs(self, model: nn.Module, loader, device):
        print(f"[BADGE] Starting feature collection with {len(loader)} batches")
        feats, probs = [], []
        use_amp = True  # set False if you prefer
        scaler_dtype = torch.float16 if getattr(self, "badge_fp16", False) else torch.float32
        print(f"[BADGE] Using AMP: {use_amp}, dtype: {scaler_dtype}")

        for batch_idx, batch in enumerate(loader):
            if batch_idx % 10 == 0:  # Print every 10th batch
                print(f"[BADGE] Processing batch {batch_idx+1}/{len(loader)}")
            
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device, non_blocking=True)
            print(f"[BADGE] Batch {batch_idx+1}: x.shape={x.shape}")

            self._penultimate_cache = None
            try:
                if use_amp:
                    with torch.autocast(device_type=device.type, dtype=scaler_dtype):
                        logits = model(x)
                        p = F.softmax(logits, dim=1)
                else:
                    logits = model(x)
                    p = F.softmax(logits, dim=1)
                print(f"[BADGE] Batch {batch_idx+1}: logits.shape={logits.shape}, p.shape={p.shape}")
            except Exception as e:
                print(f"[BADGE] ERROR in forward pass for batch {batch_idx+1}: {e}")
                raise

            z = self._penultimate_cache
            if z is None:
                print(f"[BADGE] ERROR: Hook didn't capture features for batch {batch_idx+1}")
                raise RuntimeError("BADGE hook didn't capture penultimate features.")
            z = z.flatten(1)  # (B, d)
            print(f"[BADGE] Batch {batch_idx+1}: z.shape={z.shape}")

            # Keep tensors on GPU for now
            feats.append(z)        # list of (b,d)
            probs.append(p)        # list of (b,K)

        print(f"[BADGE] Concatenating {len(feats)} feature tensors...")
        Z = torch.cat(feats, dim=0)    # (N,d) on GPU
        P = torch.cat(probs, dim=0)    # (N,K) on GPU
        print(f"[BADGE] Feature collection completed: Z.shape={Z.shape}, P.shape={P.shape}")
        return Z, P

    # ------------------------------------------------------------------ BADGE embeddings
    def _badge_embeddings(self, Z: torch.Tensor, P: torch.Tensor, yhat=None, fp16=False):
        print(f"[BADGE] Computing BADGE embeddings: Z.shape={Z.shape}, P.shape={P.shape}")
        # Z: (N,d), P: (N,K)
        N, d = Z.shape
        K = P.shape[1]
        print(f"[BADGE] N={N}, d={d}, K={K}")
        
        if yhat is None:
            yhat = torch.argmax(P, dim=1)  # (N,)
            print(f"[BADGE] Computed yhat: shape={yhat.shape}, unique values={torch.unique(yhat).tolist()}")
        else:
            print(f"[BADGE] Using provided yhat: shape={yhat.shape}")
            
        device = Z.device
        dtype = torch.float16 if fp16 else torch.float32
        print(f"[BADGE] Using dtype: {dtype}")

        # E = P[:,:,None] * Z[:,None,:]  -> (N,K,d)
        print(f"[BADGE] Computing outer product P ⊗ Z...")
        E = P.unsqueeze(-1) * Z.unsqueeze(1)   # (N,K,d)
        print(f"[BADGE] Outer product computed: E.shape={E.shape}")
        
        # subtract Z in the block of the hallucinated class
        print(f"[BADGE] Subtracting Z from hallucinated class blocks...")
        rows = torch.arange(N, device=device)
        E[rows, yhat, :] -= Z
        print(f"[BADGE] Subtraction completed")
        
        G = E.reshape(N, K * d).to(dtype, copy=False)  # (N,Kd)
        print(f"[BADGE] Reshaped to gradient embeddings: G.shape={G.shape}")

        # (optional) L2-normalize for distance stability:
        # G = torch.nn.functional.normalize(G, p=2, dim=1)
        return G

    # ------------------------------------------------------------------ k-MEANS++ seeding
    def _kmeanspp(self, G: torch.Tensor, k: int):
        """
        G: (N,D) on GPU. Returns python list of indices (length k).
        """
        print(f"[BADGE] Starting k-means++ with G.shape={G.shape}, k={k}")
        device = G.device
        N = G.shape[0]
        print(f"[BADGE] N={N}, device={device}")
        
        rng = torch.Generator(device=device)
        seed = int(getattr(self, "random_seed", 0) or 0)
        if seed:
            rng.manual_seed(seed)
            print(f"[BADGE] Set random seed: {seed}")

        # pick first center uniformly
        c0 = int(torch.randint(low=0, high=N, size=(1,), generator=rng, device=device).item())
        centers = [c0]
        print(f"[BADGE] First center: {c0}")

        # squared distances to nearest center
        # D = ||x||^2 + ||c||^2 - 2 x·c  (maintained as min over chosen centers)
        print(f"[BADGE] Computing initial distances...")
        x_norm2 = (G * G).sum(dim=1)  # (N,)
        c = G[c0]
        c_norm2 = (c * c).sum()
        D = (x_norm2 + c_norm2 - 2.0 * (G @ c))  # (N,)
        print(f"[BADGE] Initial distances computed: D.shape={D.shape}, min={D.min().item():.4f}, max={D.max().item():.4f}")

        # iterate
        for i in range(1, k):
            print(f"[BADGE] Iteration {i+1}/{k}: selecting center {i+1}")
            probs = D.clamp_min_(1e-12) / (D.sum() + 1e-12)
            print(f"[BADGE] Probabilities computed: min={probs.min().item():.6f}, max={probs.max().item():.6f}")
            
            next_idx = int(torch.multinomial(probs, 1, generator=rng).item())
            centers.append(next_idx)
            print(f"[BADGE] Selected center {i+1}: {next_idx}")

            c = G[next_idx]
            c_norm2 = (c * c).sum()
            # update min distances
            D = torch.minimum(D, x_norm2 + c_norm2 - 2.0 * (G @ c))
            print(f"[BADGE] Updated distances: min={D.min().item():.4f}, max={D.max().item():.4f}")

        print(f"[BADGE] k-means++ completed. Selected centers: {centers}")
        return centers

    # ------------------------------------------------------------------ Model hook helpers
    def _attach_last_linear_input_hook(self, model: nn.Module) -> None:
        """
        Attach a forward hook on the last Linear layer so we can capture *its input* (penultimate features).
        """
        last_linear = self._find_last_linear(model)
        assert isinstance(last_linear, nn.Linear), "Could not find final nn.Linear layer for BADGE."
        self._last_linear = last_linear

        def _hook(module, inputs, output):
            # inputs is a tuple; inputs[0] is (B, d) penultimate features (possibly (B, d, 1, 1))
            x = inputs[0]
            self._penultimate_cache = x

        self._hook_handle = last_linear.register_forward_hook(_hook)

    def _detach_hook(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def _find_last_linear(self, model: nn.Module) -> nn.Module:
        """
        Try common classifier attributes first (ResNet/DenseNet), then fall back to last nn.Linear in the module tree.
        """
        # Common classifier heads
        for name in ["fc", "classifier", "head", "last_linear"]:
            m = getattr(model, name, None)
            if isinstance(m, nn.Linear):
                return m
            # DenseNet sometimes has Sequential classifier; grab last Linear within it
            if isinstance(m, nn.Sequential):
                for sub in reversed(list(m.modules())):
                    if isinstance(sub, nn.Linear):
                        return sub

        # Fallback: last nn.Linear anywhere in the model
        last_linear = None
        for sub in model.modules():
            if isinstance(sub, nn.Linear):
                last_linear = sub
        if last_linear is None:
            raise RuntimeError("BADGE: could not locate a final nn.Linear layer.")
        return last_linear

    # ------------------------------------------------------------------ Pool & config helpers
    def _resolve_pool(self):
        """
        Returns (pool_dataset, pool_indices_list)
        """
        ds = (
            getattr(self, "pool_dataset", None)
            or getattr(self, "unlabeled_dataset", None)
            or getattr(self, "pool_set", None)
        )
        assert ds is not None, "BADGESampler expects a pool/unlabeled dataset."

        idxs = (
            getattr(self, "pool_indices", None)
            or getattr(self, "unlabeled_indices", None)
            or (ds.indices if isinstance(ds, Subset) else list(range(len(ds))))
        )
        return ds, list(idxs)

    def _eval_bs(self) -> int:
        return int(
            getattr(self, "eval_batch_size", None)
            or getattr(self, "batch_size", 64)
        )
