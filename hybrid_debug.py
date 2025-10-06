import os
import json
import random
import traceback

import torch
import matplotlib.pyplot as plt

# --- import your project classes ---
# Adjust these imports to match your repo structure
from active_learning_models import (
    HybridBADGECoreSetActiveLearning,  # the new hybrid class
)

def plot_results(pipeline):
    """
    Plot the results of the active learning pipeline.
    """
    print(f"[DEBUG] Generating plots...")
    if not getattr(pipeline, "accuracy_scores", None) or not getattr(pipeline, "recall_scores", None):
        print("[DEBUG] Nothing to plot (no scores on pipeline).")
        return

    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    xs = range(1, len(pipeline.accuracy_scores) + 1)
    plt.plot(list(xs), pipeline.accuracy_scores, 'o-')
    plt.title('Hybrid Sampler - Accuracy vs Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)

    # Recall
    plt.subplot(1, 2, 2)
    xs = range(1, len(pipeline.recall_scores) + 1)
    plt.plot(list(xs), pipeline.recall_scores, 'o-')
    plt.title('Hybrid Sampler - Recall vs Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Recall (%)')
    plt.grid(True)

    plt.tight_layout()
    out_png = 'hybrid_sampler_results.png'
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f"[DEBUG] Plot saved to '{out_png}'")
    plt.show()


def hybrid_debug():
    """
    Main function to run the HYBRID (BADGE + CoreSet) sampler experiment with debug monitoring.
    """
    print(f"[DEBUG] ===== Starting HYBRID Sampler Debug Run =====")

    # --------------------
    # Configuration
    # --------------------
    dataset_path     = "nih_chest_xrays_light"  # adjust if needed
    batch_size       = 32
    epochs_per_iter  = 2
    iterations       = 2
    budget_per_iter  = 1000
    test_sample_size = 1000
    seed             = 42

    # Hybrid-specific knobs
    hybrid_params = {
        "badge_ratio":        0.5,      # 50% BADGE, 50% CoreSet
        "badge_subsample":    20000,    # BADGE candidate subsample (None to disable)
        "badge_fp16":         False,    # BADGE half precision for embeddings
        "coreset_subsample":  20000,    # CoreSet candidate subsample (None to disable)
        "l2_norm":            True,     # L2-normalize features for CoreSet distances
        "dist_chunk":         2048,     # CoreSet chunk size for cdist/min-dist
    }
    model_name = "resnet18"  # or "resnet50", "densenet121"

    print(f"[DEBUG] Configuration:")
    print(f"[DEBUG]   - dataset_path: {dataset_path}")
    print(f"[DEBUG]   - batch_size: {batch_size}")
    print(f"[DEBUG]   - epochs_per_iter: {epochs_per_iter}")
    print(f"[DEBUG]   - iterations: {iterations}")
    print(f"[DEBUG]   - budget_per_iter: {budget_per_iter}")
    print(f"[DEBUG]   - test_sample_size: {test_sample_size}")
    print(f"[DEBUG]   - seed: {seed}")
    print(f"[DEBUG]   - model_name: {model_name}")
    print(f"[DEBUG] Hybrid params: {hybrid_params}")

    # --------------------
    # Device setup
    # --------------------
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

    # Seed for debugging reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --------------------
    # Initialize pipeline
    # --------------------
    print(f"\n[DEBUG] Initializing HYBRID Sampler Pipeline...")
    try:
        pipe = HybridBADGECoreSetActiveLearning(
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
            **hybrid_params
        )
        print(f"[DEBUG] HYBRID pipeline initialized successfully")
    except Exception as e:
        print(f"[DEBUG] ERROR initializing pipeline: {e}")
        traceback.print_exc()
        return

    # --------------------
    # Run pipeline
    # --------------------
    print(f"\n[DEBUG] Starting HYBRID Active Learning Pipeline...")
    print(f"[DEBUG] " + "=" * 60)
    try:
        accuracy_scores, recall_scores = pipe.run_pipeline()
        pipe.accuracy_scores = accuracy_scores
        pipe.recall_scores   = recall_scores
        print(f"[DEBUG] Pipeline completed successfully!")
    except Exception as e:
        print(f"[DEBUG] ERROR in pipeline execution: {e}")
        traceback.print_exc()
        return

    # --------------------
    # Print final results
    # --------------------
    if not accuracy_scores or not recall_scores:
        print("[DEBUG] No scores returned from pipeline.")
        return

    print(f"\n[DEBUG] " + "=" * 60)
    print(f"[DEBUG] FINAL RESULTS:")
    print(f"[DEBUG] Final Accuracy: {accuracy_scores[-1]:.2f}%")
    print(f"[DEBUG] Final Recall: {recall_scores[-1]:.2f}%")
    print(f"[DEBUG] Best Accuracy: {max(accuracy_scores):.2f}%")
    print(f"[DEBUG] Best Recall: {max(recall_scores):.2f}%")

    # --------------------
    # Plot & Save
    # --------------------
    print(f"\n[DEBUG] Generating plots...")
    try:
        plot_results(pipe)
    except Exception as e:
        print(f"[DEBUG] ERROR generating plots: {e}")
        traceback.print_exc()

    print(f"[DEBUG] Saving results to file...")
    try:
        results_item = {
            "accuracy_scores":   accuracy_scores,
            "recall_scores":     recall_scores,
            "iterations":        iterations,
            "budget_per_iter":   budget_per_iter,
            "epochs_per_iter":   epochs_per_iter,
            "batch_size":        batch_size,
            "model_name":        model_name,
            "hybrid_params":     hybrid_params,
        }

        out_json = "hybrid_sampler_results.json"
        # Append to list of runs
        if os.path.exists(out_json):
            with open(out_json, "r") as f:
                try:
                    data = json.load(f)
                    if not isinstance(data, list):
                        data = [data]
                except json.JSONDecodeError:
                    data = []
        else:
            data = []
        data.append(results_item)
        with open(out_json, "w") as f:
            json.dump(data, f, indent=2)

        print(f"[DEBUG] Results saved to '{out_json}'")
        print(f"[DEBUG] Plot saved to 'hybrid_sampler_results.png'")
    except Exception as e:
        print(f"[DEBUG] ERROR saving results: {e}")
        traceback.print_exc()

    print(f"[DEBUG] ===== HYBRID Sampler Debug Run Completed =====")


if __name__ == "__main__":
    hybrid_debug()
