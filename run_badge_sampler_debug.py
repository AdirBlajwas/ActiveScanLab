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
import json
# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from custom_dataset import ChestXrayDataset
from classifier_models import Resnet18Model
from active_learning_models import *


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
    plt.savefig('plots/badge_sampler_results.png', dpi=300, bbox_inches='tight')
    print(f"[DEBUG] Plot saved to 'plots'")
    plt.show()

def coreset_debug():
    """
    Main function to run the CoreSet sampler experiment with debug monitoring.
    """
    import json, os, traceback
    print(f"[DEBUG] ===== Starting CoreSet Sampler Debug Run =====")

    # Configuration
    dataset_path     = "nih_chest_xrays_light"
    batch_size       = 32
    epochs_per_iter  = 2
    iterations       = 2
    budget_per_iter  = 5000
    test_sample_size = 1000
    seed             = 42
    model_name       = 'resnet18'

    print(f"[DEBUG] Configuration:")
    print(f"[DEBUG]   - dataset_path: {dataset_path}")
    print(f"[DEBUG]   - batch_size: {batch_size}")
    print(f"[DEBUG]   - epochs_per_iter: {epochs_per_iter}")
    print(f"[DEBUG]   - iterations: {iterations}")
    print(f"[DEBUG]   - budget_per_iter: {budget_per_iter}")
    print(f"[DEBUG]   - test_sample_size: {test_sample_size}")
    print(f"[DEBUG]   - seed: {seed}")
    print(f"[DEBUG]   - model_name: {model_name}")

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

    # CoreSet-specific parameters (match your CoreSetSamplingActiveLearning __init__ signature)
    coreset_params = {
        "coreset_subsample": 50000,  # optionally subsample large pools for speed
        "l2_norm": True,             # L2-normalize penultimate features before distances
        "dist_chunk": 2048,          # chunk size for min-distance computation
    }
    print(f"[DEBUG] CoreSet parameters: {coreset_params}")

    # Initialize CoreSet pipeline
    print(f"\n[DEBUG] Initializing CoreSet Sampler Pipeline...")
    try:
        coreset_pipeline = CoreSetSamplingActiveLearning(
            device=device,
            iterations=iterations,
            epochs_per_iter=epochs_per_iter,
            budget_per_iter=budget_per_iter,
            model_name=model_name,
            objective_function_name='BCEWithLogitsLoss',
            optimizer_name='Adam',
            root_dir=dataset_path,
            batch_size=batch_size,
            test_sample_size=test_sample_size,
            seed=seed,
            **coreset_params
        )
        print(f"[DEBUG] CoreSet pipeline initialized successfully")
    except Exception as e:
        print(f"[DEBUG] ERROR initializing pipeline: {e}")
        traceback.print_exc()
        return

    # Run the pipeline
    print(f"\n[DEBUG] Starting CoreSet Active Learning Pipeline...")
    print(f"[DEBUG] " + "=" * 60)
    try:
        accuracy_scores, recall_scores = coreset_pipeline.run_pipeline()
        coreset_pipeline.accuracy_scores = accuracy_scores
        coreset_pipeline.recall_scores   = recall_scores
        print(f"[DEBUG] Pipeline completed successfully!")
    except Exception as e:
        print(f"[DEBUG] ERROR in pipeline execution: {e}")
        traceback.print_exc()
        return

    # Print final results
    if not accuracy_scores or not recall_scores:
        print("[DEBUG] No scores returned from pipeline.")
        return
    print(f"\n[DEBUG] " + "=" * 60)
    print(f"[DEBUG] FINAL RESULTS:")
    print(f"[DEBUG] Final Accuracy: {accuracy_scores[-1]:.2f}%")
    print(f"[DEBUG] Final Recall:   {recall_scores[-1]:.2f}%")
    print(f"[DEBUG] Best Accuracy:  {max(accuracy_scores):.2f}%")
    print(f"[DEBUG] Best Recall:    {max(recall_scores):.2f}%")

    # Plot results (re-uses your existing plot_results(pipeline))
    print(f"\n[DEBUG] Generating plots...")
    try:
        plot_results(coreset_pipeline)
    except Exception as e:
        print(f"[DEBUG] ERROR generating plots: {e}")
        traceback.print_exc()

    # Save results to file (append-friendly)
    print(f"[DEBUG] Saving results to file...")
    try:
        results = {
            "strategy": "coreset",
            "model_name": model_name,
            "accuracy_scores": accuracy_scores,
            "recall_scores": recall_scores,
            "iterations": iterations,
            "budget_per_iter": budget_per_iter,
            "epochs_per_iter": epochs_per_iter,
            "batch_size": batch_size,
            "coreset_params": coreset_params,
            "seed": seed,
        }

        out_path = "coreset_sampler_results.json"
        if os.path.exists(out_path):
            try:
                with open(out_path, "r") as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    data = [data]
            except json.JSONDecodeError:
                data = []
        else:
            data = []
        data.append(results)
        with open(out_path, "w") as f:
            json.dump(data, f, indent=4)

        print(f"[DEBUG] Results saved to '{out_path}'")
        print(f"[DEBUG] Plot saved to 'badge_sampler_results.png' (same plotting fn)")
    except Exception as e:
        print(f"[DEBUG] ERROR saving results: {e}")
        traceback.print_exc()

    print(f"[DEBUG] ===== CoreSet Sampler Debug Run Completed =====")


def badge_debug():
    """
    Main function to run the BADGE sampler experiment with debug monitoring.
    """
    print(f"[DEBUG] ===== Starting BADGE Sampler Debug Run =====")
    
    # Configuration
    dataset_path = "nih_chest_xrays_light"
    batch_size = 32
    epochs_per_iter = 80
    iterations = 10
    budget_per_iter = 5000
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
        'badge_subsample': 50000,  # Set to a number to subsample large pools
        'badge_fp16': False,  # Use fp16 for memory efficiency
    }
    model_name = 'resnet18'
    print(f"[DEBUG] BADGE parameters: {badge_params}")
    
    # Initialize BADGE pipeline
    print(f"\n[DEBUG] Initializing BADGE Sampler Pipeline...")
    try:
        badge_pipeline = BADGESamplingActiveLearning(
            device=device,
            iterations=iterations,
            epochs_per_iter=epochs_per_iter,
            budget_per_iter=budget_per_iter,
            model_name=model_name,
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
        badge_pipeline.accuracy_scores = accuracy_scores
        badge_pipeline.recall_scores   = recall_scores
        print(f"[DEBUG] Pipeline completed successfully!")
    except Exception as e:
        print(f"[DEBUG] ERROR in pipeline execution: {e}")
        traceback.print_exc()
        return
    
    # Print final results
    if not accuracy_scores or not recall_scores:
        print("[DEBUG] No scores returned from pipeline.")
        return
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
            'badge_params': badge_params,
            'model_name': model_name
            }
        
        import json
        with open('badge_sampler_results.jso', "r") as f:
            try:
              data = json.load(f)
            except json.JSONDecodeError:
              data = []
        data.append(results)
        with open('badge_sampler_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"[DEBUG] Results saved to 'badge_sampler_results.json'")
        print(f"[DEBUG] Plot saved to 'badge_sampler_results.png'")
    except Exception as e:
        print(f"[DEBUG] ERROR saving results: {e}")
        traceback.print_exc()
    
    print(f"[DEBUG] ===== BADGE Sampler Debug Run Completed =====")


if __name__ == "__main__":
    # badge_debug()
    coreset_debug()
