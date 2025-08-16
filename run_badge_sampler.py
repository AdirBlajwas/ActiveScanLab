#!/usr/bin/env python3
"""
Simple BADGE Sampler Runner Script

This script runs the BADGE sampler using the existing BADGESampler class
from badge_sample.py integrated with the active learning pipeline.
"""

import os
import sys
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from custom_dataset import ChestXrayDataset
from classifier_models import Resnet18Model
from active_learning_models import ActiveLearningPipeline
from badge_sample import BADGESampler


class BADGESamplerPipeline(ActiveLearningPipeline):
    """
    Active Learning Pipeline that uses BADGE sampling.
    Inherits from ActiveLearningPipeline and overrides the sampling method.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize BADGE sampler
        self.badge_sampler = BADGESampler()
        
        # Set BADGE-specific parameters
        self.badge_sampler.badge_subsample = kwargs.get('badge_subsample', None)
        self.badge_sampler.badge_chunk_size = kwargs.get('badge_chunk_size', None)
        self.badge_sampler.badge_use_memmap = kwargs.get('badge_use_memmap', False)
        self.badge_sampler.badge_fp16 = kwargs.get('badge_fp16', True)
        
        # Set random seed for reproducibility
        if self.seed is not None:
            torch.manual_seed(self.seed)
            random.seed(self.seed)
            np.random.seed(self.seed)
    
    def _sampling(self, model=None):
        """
        Override the sampling method to use BADGE sampling.
        """
        print(f"[PIPELINE] _sampling called with model={type(model).__name__ if model else None}")
        
        if model is None:
            # For the first iteration, use random sampling
            print(f"[PIPELINE] First iteration - using random sampling for {self.budget_per_iter} samples")
            return random.sample(list(self.pool_indices), self.budget_per_iter)
        
        print(f"[PIPELINE] Setting up BADGE sampler...")
        # Set up BADGE sampler with current model and pool
        self.badge_sampler.model = model
        self.badge_sampler.device = self.device
        self.badge_sampler.pool_dataset = self.dataset
        self.badge_sampler.pool_indices = list(self.pool_indices)
        self.badge_sampler.eval_batch_size = self.batch_size
        self.badge_sampler.random_seed = self.seed
        
        print(f"[PIPELINE] BADGE sampler configured:")
        print(f"[PIPELINE]   - model: {type(self.badge_sampler.model).__name__}")
        print(f"[PIPELINE]   - device: {self.badge_sampler.device}")
        print(f"[PIPELINE]   - pool_size: {len(self.badge_sampler.pool_indices)}")
        print(f"[PIPELINE]   - eval_batch_size: {self.badge_sampler.eval_batch_size}")
        
        # Run BADGE sampling
        print(f"[PIPELINE] Calling BADGE sampler._sample({self.budget_per_iter})...")
        try:
            selected_indices = self.badge_sampler._sample(self.budget_per_iter)
            print(f"[PIPELINE] BADGE sampling completed successfully, returned {len(selected_indices)} indices")
            return selected_indices
        except Exception as e:
            print(f"[PIPELINE] ERROR in BADGE sampling: {e}")
            import traceback
            traceback.print_exc()
            raise


def plot_results(pipeline):
    """
    Plot the results of the active learning pipeline.
    """
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(pipeline.accuracy_scores) + 1), pipeline.accuracy_scores, 'b-o')
    plt.title('BADGE Sampler - Accuracy vs Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    
    # Plot recall
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(pipeline.recall_scores) + 1), pipeline.recall_scores, 'r-o')
    plt.title('BADGE Sampler - Recall vs Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Recall (%)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('badge_sampler_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """
    Main function to run the BADGE sampler experiment.
    """
    # Configuration
    dataset_path = "nih_chest_xrays_light"
    batch_size = 32
    epochs_per_iter = 3
    iterations = 10
    budget_per_iter = 100
    test_sample_size = 1000
    seed = 42
    
    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    print(f"Dataset path: {dataset_path}")
    
    # BADGE-specific parameters
    badge_params = {
        'badge_subsample': None,  # Set to a number to subsample large pools
        'badge_chunk_size': None,  # Set to control inference chunk size
        'badge_use_memmap': False,  # Use memmap for large distance matrices
        'badge_fp16': True,  # Use fp16 for memory efficiency
    }
    
    # Initialize BADGE pipeline
    print("Initializing BADGE Sampler Pipeline...")
    badge_pipeline = BADGESamplerPipeline(
        device=device,
        iterations=iterations,
        epochs_per_iter=epochs_per_iter,
        budget_per_iter=budget_per_iter,
        model_name='resnet18',
        objective_function_name='BCEWithLogitsLoss',
        optimizer_name='Adam',
        root_dir=dataset_path,
        batch_size=batch_size,
        test_sample_size=test_sample_size,
        seed=seed,
        **badge_params
    )
    
    # Run the pipeline
    print("Starting BADGE Active Learning Pipeline...")
    print("=" * 60)
    
    accuracy_scores, recall_scores = badge_pipeline.run_pipeline()
    
    # Print final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS:")
    print(f"Final Accuracy: {accuracy_scores[-1]:.2f}%")
    print(f"Final Recall: {recall_scores[-1]:.2f}%")
    print(f"Best Accuracy: {max(accuracy_scores):.2f}%")
    print(f"Best Recall: {max(recall_scores):.2f}%")
    
    # Plot results
    print("\nGenerating plots...")
    plot_results(badge_pipeline)
    
    # Save results to file
    results = {
        'accuracy_scores': accuracy_scores,
        'recall_scores': recall_scores,
        'iterations': iterations,
        'budget_per_iter': budget_per_iter,
        'epochs_per_iter': epochs_per_iter,
        'batch_size': batch_size,
        'badge_params': badge_params
    }
    
    import json
    with open('badge_sampler_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Results saved to 'badge_sampler_results.json'")
    print("Plot saved to 'badge_sampler_results.png'")


if __name__ == "__main__":
    main()
