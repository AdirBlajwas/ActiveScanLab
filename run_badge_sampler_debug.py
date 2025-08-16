#!/usr/bin/env python3
"""
Debug BADGE Sampler Runner Script

This script runs the BADGE sampler with extensive progress monitoring
and debug prints to identify where the pipeline gets stuck.
"""

import os
import sys
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
import traceback
import time

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from custom_dataset import ChestXrayDataset
from classifier_models import Resnet18Model
from active_learning_models import ActiveLearningPipeline
from badge_sample import BADGESampler


class BADGESamplerPipeline(ActiveLearningPipeline):
    """
    Active Learning Pipeline that uses BADGE sampling with debug prints.
    """
    
    def __init__(self, *args, **kwargs):
        print(f"[DEBUG] Initializing BADGESamplerPipeline with args={len(args)}, kwargs={list(kwargs.keys())}")
        super().__init__(*args, **kwargs)
        
        # Initialize BADGE sampler
        print(f"[DEBUG] Creating BADGESampler instance...")
        self.badge_sampler = BADGESampler()
        
        # Set BADGE-specific parameters
        self.badge_sampler.badge_subsample = kwargs.get('badge_subsample', None)
        self.badge_sampler.badge_chunk_size = kwargs.get('badge_chunk_size', None)
        self.badge_sampler.badge_use_memmap = kwargs.get('badge_use_memmap', False)
        self.badge_sampler.badge_fp16 = kwargs.get('badge_fp16', True)
        
        print(f"[DEBUG] BADGE parameters set:")
        print(f"[DEBUG]   - badge_subsample: {self.badge_sampler.badge_subsample}")
        print(f"[DEBUG]   - badge_chunk_size: {self.badge_sampler.badge_chunk_size}")
        print(f"[DEBUG]   - badge_use_memmap: {self.badge_sampler.badge_use_memmap}")
        print(f"[DEBUG]   - badge_fp16: {self.badge_sampler.badge_fp16}")
        
        # Set random seed for reproducibility
        if self.seed is not None:
            print(f"[DEBUG] Setting random seeds to {self.seed}")
            torch.manual_seed(self.seed)
            random.seed(self.seed)
            np.random.seed(self.seed)
        
        print(f"[DEBUG] BADGESamplerPipeline initialization completed")
    
    def _sampling(self, model=None):
        """
        Override the sampling method to use BADGE sampling with debug prints.
        """
        print(f"\n[DEBUG] ===== _sampling called =====")
        print(f"[DEBUG] model type: {type(model).__name__ if model else 'None'}")
        print(f"[DEBUG] budget_per_iter: {self.budget_per_iter}")
        print(f"[DEBUG] pool_indices size: {len(self.pool_indices)}")
        
        if model is None:
            # For the first iteration, use random sampling
            print(f"[DEBUG] First iteration - using random sampling")
            selected = random.sample(list(self.pool_indices), self.budget_per_iter)
            print(f"[DEBUG] Random sampling completed: {len(selected)} samples")
            return selected
        
        print(f"[DEBUG] Setting up BADGE sampler...")
        start_time = time.time()
        
        # Set up BADGE sampler with current model and pool
        self.badge_sampler.model = model
        self.badge_sampler.device = self.device
        self.badge_sampler.pool_dataset = self.dataset
        self.badge_sampler.pool_indices = list(self.pool_indices)
        self.badge_sampler.eval_batch_size = self.batch_size
        self.badge_sampler.random_seed = self.seed
        
        setup_time = time.time() - start_time
        print(f"[DEBUG] BADGE sampler setup completed in {setup_time:.2f}s")
        print(f"[DEBUG] BADGE sampler configuration:")
        print(f"[DEBUG]   - model: {type(self.badge_sampler.model).__name__}")
        print(f"[DEBUG]   - device: {self.badge_sampler.device}")
        print(f"[DEBUG]   - pool_size: {len(self.badge_sampler.pool_indices)}")
        print(f"[DEBUG]   - eval_batch_size: {self.badge_sampler.eval_batch_size}")
        print(f"[DEBUG]   - random_seed: {self.badge_sampler.random_seed}")
        
        # Run BADGE sampling
        print(f"[DEBUG] Calling BADGE sampler._sample({self.budget_per_iter})...")
        sampling_start = time.time()
        
        try:
            selected_indices = self.badge_sampler._sample(self.budget_per_iter)
            sampling_time = time.time() - sampling_start
            print(f"[DEBUG] BADGE sampling completed successfully in {sampling_time:.2f}s")
            print(f"[DEBUG] Returned {len(selected_indices)} indices")
            print(f"[DEBUG] Sample indices: {selected_indices[:5]}...")
            return selected_indices
        except Exception as e:
            print(f"[DEBUG] ERROR in BADGE sampling: {e}")
            print(f"[DEBUG] Full traceback:")
            traceback.print_exc()
            raise


def plot_results(pipeline):
    """
    Plot the results of the active learning pipeline.
    """
    print(f"[DEBUG] Generating plots...")
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
    print(f"[DEBUG] Plot saved to 'badge_sampler_results.png'")
    plt.show()


def main():
    """
    Main function to run the BADGE sampler experiment with debug monitoring.
    """
    print(f"[DEBUG] ===== Starting BADGE Sampler Debug Run =====")
    
    # Configuration
    dataset_path = "nih_chest_xrays_light"
    batch_size = 32
    epochs_per_iter = 3
    iterations = 10
    budget_per_iter = 100
    test_sample_size = 1000  # Using test sample size for debugging
    seed = 42
    
    print(f"[DEBUG] Configuration:")
    print(f"[DEBUG]   - dataset_path: {dataset_path}")
    print(f"[DEBUG]   - batch_size: {batch_size}")
    print(f"[DEBUG]   - epochs_per_iter: {epochs_per_iter}")
    print(f"[DEBUG]   - iterations: {iterations}")
    print(f"[DEBUG]   - budget_per_iter: {budget_per_iter}")
    print(f"[DEBUG]   - test_sample_size: {test_sample_size}")
    print(f"[DEBUG]   - seed: {seed}")
    
    # Device setup
    print(f"[DEBUG] Setting up device...")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"[DEBUG] Using device: {device}")
    print(f"[DEBUG] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[DEBUG] CUDA device count: {torch.cuda.device_count()}")
        print(f"[DEBUG] Current CUDA device: {torch.cuda.current_device()}")
    
    # BADGE-specific parameters
    badge_params = {
        'badge_subsample': None,  # Set to a number to subsample large pools
        'badge_chunk_size': None,  # Set to control inference chunk size
        'badge_use_memmap': False,  # Use memmap for large distance matrices
        'badge_fp16': True,  # Use fp16 for memory efficiency
    }
    
    print(f"[DEBUG] BADGE parameters: {badge_params}")
    
    # Initialize BADGE pipeline
    print(f"\n[DEBUG] Initializing BADGE Sampler Pipeline...")
    try:
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
        print(f"[DEBUG] BADGE pipeline initialized successfully")
    except Exception as e:
        print(f"[DEBUG] ERROR initializing pipeline: {e}")
        traceback.print_exc()
        return
    
    # Run the pipeline
    print(f"\n[DEBUG] Starting BADGE Active Learning Pipeline...")
    print(f"[DEBUG] " + "=" * 60)
    
    try:
        accuracy_scores, recall_scores = badge_pipeline.run_pipeline()
        print(f"[DEBUG] Pipeline completed successfully!")
    except Exception as e:
        print(f"[DEBUG] ERROR in pipeline execution: {e}")
        traceback.print_exc()
        return
    
    # Print final results
    print(f"\n[DEBUG] " + "=" * 60)
    print(f"[DEBUG] FINAL RESULTS:")
    print(f"[DEBUG] Final Accuracy: {accuracy_scores[-1]:.2f}%")
    print(f"[DEBUG] Final Recall: {recall_scores[-1]:.2f}%")
    print(f"[DEBUG] Best Accuracy: {max(accuracy_scores):.2f}%")
    print(f"[DEBUG] Best Recall: {max(recall_scores):.2f}%")
    
    # Plot results
    print(f"\n[DEBUG] Generating plots...")
    try:
        plot_results(badge_pipeline)
    except Exception as e:
        print(f"[DEBUG] ERROR generating plots: {e}")
        traceback.print_exc()
    
    # Save results to file
    print(f"[DEBUG] Saving results to file...")
    try:
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
        
        print(f"[DEBUG] Results saved to 'badge_sampler_results.json'")
        print(f"[DEBUG] Plot saved to 'badge_sampler_results.png'")
    except Exception as e:
        print(f"[DEBUG] ERROR saving results: {e}")
        traceback.print_exc()
    
    print(f"[DEBUG] ===== BADGE Sampler Debug Run Completed =====")


if __name__ == "__main__":
    main()
