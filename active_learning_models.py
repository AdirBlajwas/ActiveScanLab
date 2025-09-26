from custom_dataset import ChestXrayDataset
from classifier_models import Resnet18Model, Resnet50Model, Densenet121Model
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import random
from typing import Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import os
import pickle
class ActiveLearningPipeline:
    def __init__(self, 
                 device,
                 iterations,
                 epochs_per_iter,
                 budget_per_iter,
                 model_name: str,
                 objective_function_name: str = 'BCEWithLogitsLoss',
                 optimizer_name: str = 'Adam',
                 lr = 1e-2,
                 root_dir = None,
                 dataset=None,
                 batch_size=32,
                 test_sample_size=None,
                 seed=42,
                 max_train_size=60000,
                 resume_iter: int = None): #NOTE: update default values later as needed

        if dataset is None and root_dir is None:
            raise ValueError("Either dataset or root_dir should be provided")
        self.seed = seed
        random.seed(self.seed)
        self.iterations = iterations
        self.budget_per_iter = budget_per_iter
        self.batch_size = batch_size
        self.max_train_size = max_train_size
        self.epochs_per_iter = epochs_per_iter
        self.root_dir = root_dir
        self.dataset = ChestXrayDataset(self.root_dir, split_type='from_files') if dataset is None else dataset
        self.pool_loader = self.dataset.get_dataloader(from_split='train', batch_size=self.batch_size, shuffle=False)
        if not resume_iter:
            self.pool_indices = set(self.dataset.train_indices)
            # Initialize train_indices with a random 10% of the pool as the initial labeled set
            total_pool = list(self.pool_indices)
            initial_train_size = max(1, int(0.1 * len(total_pool)))
            self.train_indices = set(random.sample(total_pool, initial_train_size))
            self._update_pool_indices(self.train_indices)
            self.start_iteration = 0
        else:
            print(f"Resuming from iteration {resume_iter}, loading previous train and pool indices from files")
            self.train_indices, self.pool_indices = self._load_checkpoint(resume_iter)
            self.start_iteration = resume_iter + 1
        
        self.test_indices = self.dataset.test_indices
        if test_sample_size is not None:
            self.test_indices = random.sample(self.test_indices, test_sample_size)

        self.test_loader = self.dataset.get_dataloader(from_split='test', indices=list(self.test_indices), batch_size=self.batch_size)

        
        self.device = device
        self.model_name = model_name
        self.objective_function_name = objective_function_name
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.model = self._create_classifier_model()
        
        print(f"Initializing active learning pipeline with {self.model_name} model")
        print(f"{self.epochs_per_iter} epochs per iteration, {self.iterations} iterations and {self.budget_per_iter} budget per iteration.")

        
        # test_df = self.dataset[self.dataset['Image Index'].isin(self.test_indices)] 
        # test_dataset = ChestXrayDataset(test_df, self.root_dir)
        # self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)
    def _save_checkpoint(self, iteration):
        """Save train and pool indices at the end of each iteration."""
        os.makedirs("checkpoints", exist_ok=True)
        checkpoint_path = f"checkpoints/iteration_{iteration}.pkl"
        with open(checkpoint_path, "wb") as f:
            pickle.dump({
                "train_indices": self.train_indices,
                "pool_indices": self.pool_indices
            }, f)
        print(f"Saved checkpoint for iteration {iteration} to {checkpoint_path}")

    def _load_checkpoint(self, iteration):
        """Load train and pool indices from the latest checkpoint."""
        checkpoint_path = f"checkpoints/iteration_{iteration}.pkl"
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, "rb") as f:
                data = pickle.load(f)
            print(f"Loaded checkpoint from iteration {iteration}")
            return data["train_indices"], data["pool_indices"]
        return None, None
      
    def run_pipeline(self):
        self.accuracy_scores = []
        self.recall_scores = []
        print(f"Running active learning pipeline whith {self.model_name} model")
        for iteration in range(self.start_iteration, self.iterations):
            if len(self.train_indices) > self.max_train_size:
                raise ValueError("The train set is larger than 60000 samples")

            print(f"Iteration {iteration + 1}/{self.iterations}")
            print("Train indices: ", len(self.train_indices))
            print("Pool indices: ", len(self.pool_indices))
            print("Test indices: ", len(self.test_indices))

            print("Training model")
            self._train_model()
            print("Evaluating model")
            accuracy, recall = self._evaluate_model()
            self.accuracy_scores.append(accuracy)
            self.recall_scores.append(recall)

            if len(self.pool_indices) < self.budget_per_iter:
                print("Not enough samples in pool to continue.")
                break
            print("Sampling new indices")
            new_selected_indices = self._sampling()

            self._update_train_indices(new_selected_indices)
            self._update_pool_indices(new_selected_indices)
            self._save_checkpoint(iteration)

            print(f"Accuracy: {accuracy:.4f}")
            print(f"Recall: {recall:.4f}")
            print("----------------------------------------")

        return self.accuracy_scores, self.recall_scores

    def _create_classifier_model(self):
        if self.model_name == 'resnet18':
            return Resnet18Model(optimizer=self.optimizer_name, loss_function=self.objective_function_name, lr=self.lr)
        elif self.model_name == 'resnet50':
            return Resnet50Model(optimizer=self.optimizer_name, loss_function=self.objective_function_name, lr=self.lr)
        elif self.model_name == 'densenet121':
            return Densenet121Model(optimizer=self.optimizer_name, loss_function=self.objective_function_name, lr=self.lr)
        else:
            raise ValueError(f"Model {self.model_name} not found")
    
    def _train_model(self):
        self.model.reset_classifier_head()
        train_loader = self.dataset.get_dataloader(from_split='train', indices=list(self.train_indices), batch_size=self.batch_size)
        self.model.train_model(self.device, train_loader, epochs=self.epochs_per_iter)
        

    def _evaluate_model(self):
        return self.model.evaluate(self.device, self.test_loader)

    def _sampling(self, **kwargs):
        raise NotImplementedError("Subclass should implement this method")

    
    def _update_train_indices(self, new_selected_samples):
        """
           Update the train indices by adding newly selected samples.
           new_selected_samples should be a set of image filenames.
        """
        self.train_indices.update(new_selected_samples)
        
    def _update_pool_indices(self, new_selected_samples):
        """
           Update the pool indices by removing the newly selected samples.
           new_selected_samples should be a set of image filenames.
        """
        self.pool_indices.difference_update(new_selected_samples)



class RandomSamplingActiveLearning(ActiveLearningPipeline):
    def _sampling(self, **kwargs):
        random.seed(self.seed)
        # Convert set to list for random.sample, then convert back to set
        pool_list = list(self.pool_indices)
        return set(random.sample(pool_list, self.budget_per_iter))


class BADGESamplingActiveLearning(ActiveLearningPipeline):
    def __init__(self, *args, badge_subsample: Optional[int] = None, badge_fp16: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.badge_subsample = badge_subsample   # <-- use the params you received
        self.badge_fp16 = badge_fp16
        
    def _sampling(self, **kwargs):
        """
            BADGE = Batch Active learning by Diverse Gradient Embeddings
            May improve active learning where:
            You want both uncertainty (picking hard-to-classify points).
            And diversity (picking a wide range of points, not duplicates).
            The key insight is:
            Instead of sampling based on just predictions or just diversity in feature space, sample based on the gradients you would get if you trained on the sample. 
        """
        budget = self.budget_per_iter
        print(f"[BADGE] Starting BADGE sampling with budget={budget}")
        model = self.model
        print(f"[BADGE] Model type: {type(model).__name__}")
        model.eval()
        n_pool = len(self.pool_indices)
    
        # --- Decide active pool (optionally subsample for efficiency)
        full_pool = list(self.pool_indices)  # absolute indices
        n_pool = len(full_pool)
        
        print(f"[BADGE] Full pool size: {n_pool}, badge_subsample={self.badge_subsample}")
        if self.badge_subsample and self.badge_subsample < n_pool:
            print(f"[BADGE] Subsampling pool from {n_pool} to {self.badge_subsample}")
            seed = getattr(self, "random_seed", None)
            if seed is not None:
                random.seed(int(seed))
            # sample FROM the existing pool indices
            perm = random.sample(range(n_pool), self.badge_subsample)
            active_pool = [full_pool[i] for i in perm]
        else:
            active_pool = full_pool
            
        pool_loader = self.dataset.get_dataloader(from_split='train', indices=active_pool, batch_size=self.batch_size)
        
        Z_list, P_list, ordered_pool = [], [], []
        on_device = torch.device(self.device)
        
        print(f"[BADGE] Collecting features and probabilities...")
        with torch.inference_mode():
            for imgs, _, batch_indices in pool_loader:
                imgs = imgs.to(on_device, non_blocking=True)

                # Expect your model wrapper to support return_features=True
                logits, features = model.forward(imgs, return_features=True)  # logits: (B,K or 1), features: (B,d)
                z = features.flatten(1)

                # Build probs matrix P: (B,K)
                if logits.ndim == 2 and logits.size(1) == 1:
                    # Binary head -> make 2-class distribution
                    p1 = torch.sigmoid(logits)              # (B,1)
                    p = torch.cat([1 - p1, p1], dim=1)      # (B,2)
                else:
                    p = F.softmax(logits, dim=1)            # (B,K)

                # Accumulate on device (kmeans++ will run on-device)
                Z_list.append(z)                            # (B,d)
                P_list.append(p)                            # (B,K)

                # Preserve exact pool index order (don’t assume DataLoader order == input order)
                # batch_indices may be tensor; normalize to python ints
                if isinstance(batch_indices, torch.Tensor):
                    ordered_pool.extend(batch_indices.tolist())
                else:
                    ordered_pool.extend(list(batch_indices))

        if not Z_list:
            return []

        Z = torch.cat(Z_list, dim=0)                        # (N,d)
        P = torch.cat(P_list, dim=0)                        # (N,K)
        assert Z.size(0) == P.size(0) == len(ordered_pool), "Batching/order mismatch"

        # Optionally reduce dtype
        if self.badge_fp16:
            Z = Z.half()
            P = P.half()
            
        print(f"[BADGE] Collected features: Z.shape={Z.shape}, P.shape={P.shape}")
        
        # Compute BADGE gradient embeddings (N, K*d)
        print(f"[BADGE] Computing BADGE gradient embeddings...")
        G = self._badge_embeddings(Z, P, fp16=self.badge_fp16)  # (N,Kd)
        print(f"[BADGE] Gradient embeddings computed: G.shape={G.shape}")
        
        # k-MEANS++ seeding on embeddings to pick `budget` points
        k = min(int(budget), G.size(0))
        if k <= 0:
            return []
        print(f"[BADGE] Running k-means++ seeding to select {k} points...")
        chosen_local = self._kmeanspp(G, k=k)
        print(f"[BADGE] k-means++ completed, selected {len(chosen_local)} local indices")

        # Map back to pool indices
        chosen_pool_indices = [ordered_pool[i] for i in chosen_local]
        print(f"[BADGE] Mapped to pool indices, returning {len(chosen_pool_indices)} samples")
        return chosen_pool_indices

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
            # print(f"[BADGE] Iteration {i+1}/{k}: selecting center {i+1}")
            probs = D.clamp_min_(1e-12) / (D.sum() + 1e-12)
            # print(f"[BADGE] Probabilities computed: min={probs.min().item():.6f}, max={probs.max().item():.6f}")
            
            next_idx = int(torch.multinomial(probs, 1, generator=rng).item())
            centers.append(next_idx)
            # print(f"[BADGE] Selected center {i+1}: {next_idx}")

            c = G[next_idx]
            c_norm2 = (c * c).sum()
            # update min distances
            D = torch.minimum(D, x_norm2 + c_norm2 - 2.0 * (G @ c))
            # print(f"[BADGE] Updated distances: min={D.min().item():.4f}, max={D.max().item():.4f}")

        print(f"[BADGE] k-means++ completed.")
        return centers

class CoreSetSamplingActiveLearning(ActiveLearningPipeline):
    def _sampling(self, **kwargs):
        """
        Core-set (k-center greedy) active learning:
        Selects diverse points by maximizing the minimum distance to the current labeled set in embedding space.
        """
        model = kwargs["model"]
        model.eval()

        # 1. Extract embeddings for labeled set
        labeled_loader = self.dataset.get_dataloader(
            from_split="train",
            indices=list(self.train_indices),
            batch_size=self.batch_size
        )

        labeled_embeddings = []
        for imgs, _, _ in labeled_loader:
            imgs = imgs.to(self.device)
            embs = model.get_embedding(imgs).detach().cpu().numpy()
            labeled_embeddings.append(embs)
        labeled_embeddings = np.concatenate(labeled_embeddings, axis=0)  # shape: [N_labeled, D]

        # 2. Extract embeddings for unlabeled pool
        pool_loader = self.dataset.get_dataloader(
            from_split="train",
            indices=list(self.pool_indices),
            batch_size=self.batch_size
        )

        pool_embeddings = []
        pool_names = []
        for imgs, _, names in pool_loader:
            imgs = imgs.to(self.device)
            embs = model.get_embedding(imgs).detach().cpu().numpy()
            pool_embeddings.append(embs)
            pool_names.extend(names)

        pool_embeddings = np.concatenate(pool_embeddings, axis=0)  # shape: [N_pool, D]
        pool_names = np.array(pool_names)

        # 3. Run k-center greedy selection
        selected_indices = self.k_center_greedy(
            labeled_embeddings, pool_embeddings, pool_names, self.budget_per_iter
        )

        return set(selected_indices)

    def k_center_greedy(self, labeled_embs, pool_embs, pool_names, budget):
        """
        Greedy k-center selection:
        Repeatedly pick the point in the pool that is farthest from the labeled set.
        """
        if len(labeled_embs) == 0:
            # No labeled data yet — select randomly
            return np.random.choice(pool_names, size=budget, replace=False)

        selected = []
        distances = np.min(np.linalg.norm(pool_embs[:, None] - labeled_embs[None, :], axis=2), axis=1)
        for _ in range(budget):
            idx = np.argmax(distances)
            selected.append(pool_names[idx])
            new_center = pool_embs[idx]
            dists_to_new = np.linalg.norm(pool_embs - new_center, axis=1)
            distances = np.minimum(distances, dists_to_new)
        return selected

    
def plot_results(activity_sample_1: ActiveLearningPipeline, activity_sample_2: ActiveLearningPipeline = None):
    plt.figure(figsize=(10, 6))
    name1 = activity_sample_1.__class__.__name__
    plt.plot(activity_sample_1.accuracy_scores, label=f'{name1} Accuracy', color='blue')
    if activity_sample_2 is not None:
        name2 = activity_sample_2.__class__.__name__
        plt.plot(activity_sample_2.accuracy_scores, label=f'{name2} Accuracy', color='red')
    plt.legend()
    model_name = activity_sample_1.model_name
    assert model_name == activity_sample_2.model_name if activity_sample_2 is not None else model_name, "Models should be the same for comparison in both samples"
    plt.xlabel('Iteration')
    plt.ylabel('Score (%)')
    plt.title(f'Active Learning Results whith {model_name} model')
    plt.tight_layout()
    plt.savefig(f'{activity_sample_1.__class__.__name__}_{activity_sample_2.__class__.__name__ if activity_sample_2 is not None else ""}_results.png')
    plt.show()

if __name__ == "__main__":
    print("--------------------------------")
    print("Loading dataset")
    dataset_path = "nih_chest_xrays_light"
    dataset = ChestXrayDataset(dataset_path, split_type='from_files')
    print(dataset.train_df.head())
    print("--------------------------------")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(device)
    print("--------------------------------")
    print("Active Learning Pipeline")
    active_learning_pipeline = RandomSamplingActiveLearning(
        device=device,
        iterations=10,
        root_dir=dataset_path,
        epochs_per_iter=3,
        budget_per_iter=100,
        model_name='resnet18',
        objective_function_name='BCEWithLogitsLoss',
        optimizer_name='Adam',
        test_sample_size=1000,
        seed=42,
        dataset=dataset
    )
    active_learning_pipeline.run_pipeline()
    plot_results(active_learning_pipeline)