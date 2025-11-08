"""
Main Runner for Active Learning Comparison Experiments

This module orchestrates comparison experiments between Random and Hybrid (BADGE+CoreSet)
active learning samplers across multiple random seeds. It provides:
- Robust state management with checkpoint/resume capabilities
- Frequent result saving to prevent data loss
- Clear file organization for multi-seed, multi-sampler experiments
- Independent plotting functionality for analyzing results

Usage:
    python main_runner.py --mode full      # Run full experiments
    python main_runner.py --mode plot      # Only generate plots from existing results
"""

import os
import json
import random
import traceback
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import torch
import numpy as np

# Use non-interactive backend for matplotlib to avoid display issues
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from active_learning_models import (
    RandomSamplingActiveLearning,
    HybridBADGECoreSetActiveLearning,
)


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

class ExperimentConfig:
    """
    Centralized configuration for all experiment hyperparameters.

    Default values can be overridden by loading a JSON configuration file
    from the configurations/ directory.

    To use a custom configuration:
        config = ExperimentConfig.from_json("my_config")

    The JSON file should be at: configurations/my_config.json

    JSON values will override defaults (partial override supported).
    """

    # Configuration name (default or loaded from JSON)
    CONFIG_NAME = "default"

    # Dataset settings
    DATASET_PATH = "nih_chest_xrays_light"

    # Training settings
    BATCH_SIZE = 32                    # Batch size for training and evaluation
    EPOCHS_PER_ITER = 2                # Number of training epochs per AL iteration
    MODEL_NAME = "resnet18"            # Model architecture: "resnet18", "resnet50", or "densenet121"
    OPTIMIZER_NAME = "Adam"            # Optimizer: "Adam" or "SGD"
    LOSS_FUNCTION = "BCEWithLogitsLoss"  # Loss function for binary classification
    LEARNING_RATE = 1e-3               # Learning rate for optimizer

    # Active Learning settings
    ITERATIONS = 10                    # Number of AL iterations to run
    BUDGET_PER_ITER = 1000            # Number of samples to label per iteration
    TEST_SAMPLE_SIZE = 1000           # Number of test samples to evaluate on (None = use all test samples)
    MAX_TRAIN_SIZE = 60000            # Maximum training set size
    INITIAL_TRAIN_RATIO = 0.1         # Fraction of training pool to use as initial labeled set (e.g., 0.1 = 10%)

    # Random seeds for reproducibility
    SEEDS = [42, 7, 1]            # List of random seeds to run experiments with

    # Sampler configurations
    SAMPLERS_CONFIG = {
        "random": {
            "class": RandomSamplingActiveLearning,
            "params": {}  # No additional params for random sampling
        },
        "hybrid": {
            "class": HybridBADGECoreSetActiveLearning,
            "params": {
                "badge_ratio": 0.5,           # 50% BADGE, 50% CoreSet in the mix
                "badge_subsample": 20000,     # Subsample pool to this size for BADGE (None=use all)
                "badge_fp16": False,          # Use FP16 for BADGE embeddings (faster, less precise)
                "coreset_subsample": 20000,   # Subsample pool to this size for CoreSet (None=use all)
                "l2_norm": True,              # L2-normalize features for CoreSet distance computation
                "dist_chunk": 2048,           # Chunk size for CoreSet distance computation (memory/speed tradeoff)
            }
        }
    }

    # Output settings
    RESULTS_DIR = "results"            # Directory to save all results
    CHECKPOINT_DIR = "checkpoints_main"  # Directory to save experiment checkpoints
    PLOTS_DIR = "plots"                # Directory to save generated plots
    CONFIGS_DIR = "configurations"     # Directory containing JSON configuration files

    @classmethod
    def from_json(cls, config_name: str) -> 'ExperimentConfig':
        """
        Load configuration from a JSON file, overriding default values.

        Args:
            config_name: Name of the configuration file (without .json extension)
                        File should be at: configurations/<config_name>.json

        Returns:
            ExperimentConfig instance with values loaded from JSON

        Example JSON structure:
        {
            "ITERATIONS": 5,
            "BUDGET_PER_ITER": 500,
            "SEEDS": [42],
            "MODEL_NAME": "resnet50",
            "SAMPLERS_CONFIG": {
                "hybrid": {
                    "params": {
                        "badge_ratio": 0.7
                    }
                }
            }
        }
        """
        config = cls()
        config.CONFIG_NAME = config_name

        # Load JSON file
        config_path = os.path.join(config.CONFIGS_DIR, f"{config_name}.json")

        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}\n"
                f"Please create the file or use --config default"
            )

        print(f"[CONFIG] Loading configuration from: {config_path}")

        with open(config_path, "r") as f:
            config_data = json.load(f)

        # Override default values with JSON values (partial override)
        for key, value in config_data.items():
            if hasattr(config, key):
                # Special handling for SAMPLERS_CONFIG (deep merge)
                if key == "SAMPLERS_CONFIG" and isinstance(value, dict):
                    config._merge_sampler_config(value)
                else:
                    setattr(config, key, value)
                    print(f"[CONFIG] Overriding {key}: {value}")
            else:
                print(f"[CONFIG WARNING] Unknown config key '{key}' in JSON (ignored)")

        print(f"[CONFIG] Successfully loaded configuration: {config_name}")
        return config

    def _merge_sampler_config(self, json_config: dict) -> None:
        """
        Deep merge sampler configuration from JSON into existing config.
        This allows partial override of sampler parameters.

        Args:
            json_config: Sampler configuration from JSON
        """
        for sampler_name, sampler_data in json_config.items():
            if sampler_name not in self.SAMPLERS_CONFIG:
                print(f"[CONFIG WARNING] Unknown sampler '{sampler_name}' in JSON (ignored)")
                continue

            # Merge params if provided
            if "params" in sampler_data and isinstance(sampler_data["params"], dict):
                if "params" not in self.SAMPLERS_CONFIG[sampler_name]:
                    self.SAMPLERS_CONFIG[sampler_name]["params"] = {}

                self.SAMPLERS_CONFIG[sampler_name]["params"].update(sampler_data["params"])
                print(f"[CONFIG] Updated {sampler_name} params: {sampler_data['params']}")

    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary (for saving to JSON results).

        Returns:
            Dictionary with all configuration values (excluding class methods)
        """
        config_dict = {}
        for key in dir(self):
            if key.isupper() and not key.startswith("_"):
                value = getattr(self, key)
                # Skip SAMPLERS_CONFIG as it contains class references
                if key != "SAMPLERS_CONFIG":
                    config_dict[key] = value
        return config_dict


# ============================================================================
# DIRECTORY MANAGEMENT
# ============================================================================

def setup_directories(config: ExperimentConfig) -> None:
    """
    Create necessary directories for saving results, checkpoints, and plots.

    Args:
        config: Experiment configuration object
    """
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    print(f"[SETUP] Created directories: {config.RESULTS_DIR}, {config.CHECKPOINT_DIR}, {config.PLOTS_DIR}")


# ============================================================================
# FILE NAMING UTILITIES
# ============================================================================

def get_experiment_id(sampler_name: str, seed: int, config_name: str = "default") -> str:
    """
    Generate a unique identifier for an experiment.

    Args:
        sampler_name: Name of the sampling strategy (e.g., "random", "hybrid")
        seed: Random seed used for the experiment
        config_name: Name of the configuration used (default: "default")

    Returns:
        Unique experiment identifier string (e.g., "random_seed42_config_default")
    """
    if config_name == "default":
        return f"{sampler_name}_seed{seed}"
    else:
        return f"{sampler_name}_seed{seed}_config_{config_name}"


def get_results_path(config: ExperimentConfig, sampler_name: str, seed: int) -> str:
    """
    Get the file path for saving/loading experiment results.

    Args:
        config: Experiment configuration object
        sampler_name: Name of the sampling strategy
        seed: Random seed

    Returns:
        Full path to the results JSON file
    """
    exp_id = get_experiment_id(sampler_name, seed, config.CONFIG_NAME)
    return os.path.join(config.RESULTS_DIR, f"{exp_id}_results.json")


def get_checkpoint_path(config: ExperimentConfig, sampler_name: str, seed: int, iteration: int) -> str:
    """
    Get the file path for saving/loading experiment checkpoint at a specific iteration.

    Args:
        config: Experiment configuration object
        sampler_name: Name of the sampling strategy
        seed: Random seed
        iteration: AL iteration number

    Returns:
        Full path to the checkpoint file
    """
    exp_id = get_experiment_id(sampler_name, seed, config.CONFIG_NAME)
    return os.path.join(config.CHECKPOINT_DIR, f"{exp_id}_iter{iteration}.pt")


def get_latest_checkpoint_iteration(config: ExperimentConfig, sampler_name: str, seed: int) -> Optional[int]:
    """
    Find the latest checkpoint iteration for a given experiment.

    Args:
        config: Experiment configuration object
        sampler_name: Name of the sampling strategy
        seed: Random seed

    Returns:
        Latest checkpoint iteration number, or None if no checkpoints exist
    """
    exp_id = get_experiment_id(sampler_name, seed, config.CONFIG_NAME)
    checkpoints = []

    if not os.path.exists(config.CHECKPOINT_DIR):
        return None

    for filename in os.listdir(config.CHECKPOINT_DIR):
        if filename.startswith(exp_id) and filename.endswith(".pt"):
            try:
                # Extract iteration number from filename
                iter_str = filename.replace(exp_id + "_iter", "").replace(".pt", "")
                iteration = int(iter_str)
                checkpoints.append(iteration)
            except ValueError:
                continue

    return max(checkpoints) if checkpoints else None


# ============================================================================
# STATE MANAGEMENT
# ============================================================================

def save_experiment_state(
    config: ExperimentConfig,
    pipeline: Any,
    sampler_name: str,
    seed: int,
    iteration: int,
    accuracy_scores: List[float],
    recall_scores: List[float]
) -> None:
    """
    Save complete experiment state to enable resuming after interruption.

    This saves:
    - Model state dict (weights and biases)
    - Optimizer state dict
    - Train and pool indices
    - Performance metrics up to this iteration
    - Iteration number

    Args:
        config: Experiment configuration object
        pipeline: Active learning pipeline object
        sampler_name: Name of the sampling strategy
        seed: Random seed
        iteration: Current AL iteration number
        accuracy_scores: List of accuracy scores up to current iteration
        recall_scores: List of recall scores up to current iteration
    """
    checkpoint_path = get_checkpoint_path(config, sampler_name, seed, iteration)

    state = {
        "iteration": iteration,
        "seed": seed,
        "sampler_name": sampler_name,
        "model_state_dict": pipeline.model.state_dict(),
        "optimizer_state_dict": pipeline.model.optimizer.state_dict(),
        "train_indices": list(pipeline.train_indices),
        "pool_indices": list(pipeline.pool_indices),
        "accuracy_scores": accuracy_scores,
        "recall_scores": recall_scores,
        "timestamp": datetime.now().isoformat()
    }

    torch.save(state, checkpoint_path)
    print(f"[CHECKPOINT] Saved state to {checkpoint_path}")


def load_experiment_state(
    config: ExperimentConfig,
    pipeline: Any,
    sampler_name: str,
    seed: int,
    iteration: int
) -> Tuple[List[float], List[float]]:
    """
    Load experiment state from checkpoint to resume training.

    Args:
        config: Experiment configuration object
        pipeline: Active learning pipeline object (will be modified in-place)
        sampler_name: Name of the sampling strategy
        seed: Random seed
        iteration: Iteration number to load

    Returns:
        Tuple of (accuracy_scores, recall_scores) up to the loaded iteration
    """
    checkpoint_path = get_checkpoint_path(config, sampler_name, seed, iteration)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[CHECKPOINT] Loading state from {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=pipeline.device)

    # Restore pipeline state
    pipeline.model.load_state_dict(state["model_state_dict"])
    pipeline.model.optimizer.load_state_dict(state["optimizer_state_dict"])
    pipeline.train_indices = set(state["train_indices"])
    pipeline.pool_indices = set(state["pool_indices"])

    accuracy_scores = state["accuracy_scores"]
    recall_scores = state["recall_scores"]

    print(f"[CHECKPOINT] Restored state from iteration {iteration}")
    print(f"[CHECKPOINT] Train size: {len(pipeline.train_indices)}, Pool size: {len(pipeline.pool_indices)}")

    return accuracy_scores, recall_scores


# ============================================================================
# RESULTS MANAGEMENT
# ============================================================================

def save_results(
    config: ExperimentConfig,
    sampler_name: str,
    seed: int,
    accuracy_scores: List[float],
    recall_scores: List[float],
    iteration: int,
    status: str = "in_progress"
) -> None:
    """
    Save experiment results to JSON file.

    This is called frequently (after each iteration) to prevent data loss.
    The JSON file contains full experiment configuration and metrics.

    Args:
        config: Experiment configuration object
        sampler_name: Name of the sampling strategy
        seed: Random seed
        accuracy_scores: List of accuracy scores
        recall_scores: List of recall scores
        iteration: Current iteration number
        status: Experiment status ("in_progress", "completed", or "failed")
    """
    results_path = get_results_path(config, sampler_name, seed)

    results = {
        "experiment_id": get_experiment_id(sampler_name, seed, config.CONFIG_NAME),
        "sampler_name": sampler_name,
        "seed": seed,
        "config_name": config.CONFIG_NAME,
        "status": status,
        "current_iteration": iteration,
        "total_iterations": config.ITERATIONS,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model_name": config.MODEL_NAME,
            "batch_size": config.BATCH_SIZE,
            "epochs_per_iter": config.EPOCHS_PER_ITER,
            "budget_per_iter": config.BUDGET_PER_ITER,
            "test_sample_size": config.TEST_SAMPLE_SIZE,
            "learning_rate": config.LEARNING_RATE,
            "optimizer": config.OPTIMIZER_NAME,
            "loss_function": config.LOSS_FUNCTION,
        },
        "metrics": {
            "accuracy_scores": accuracy_scores,
            "recall_scores": recall_scores,
        }
    }

    # Add sampler-specific config if hybrid
    if sampler_name == "hybrid":
        results["config"]["sampler_params"] = config.SAMPLERS_CONFIG["hybrid"]["params"]

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[RESULTS] Saved to {results_path} (status: {status}, iteration: {iteration}/{config.ITERATIONS})")


def load_results(config: ExperimentConfig, sampler_name: str, seed: int) -> Optional[Dict[str, Any]]:
    """
    Load experiment results from JSON file.

    Args:
        config: Experiment configuration object
        sampler_name: Name of the sampling strategy
        seed: Random seed

    Returns:
        Dictionary containing results, or None if file doesn't exist
    """
    results_path = get_results_path(config, sampler_name, seed)

    if not os.path.exists(results_path):
        return None

    with open(results_path, "r") as f:
        return json.load(f)


def load_all_results(config: ExperimentConfig, samplers: Optional[List[str]] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load all experiment results from the results directory.

    Args:
        config: Experiment configuration object
        samplers: Optional list of sampler names to load. If None, loads all available samplers.

    Returns:
        Dictionary mapping sampler names to lists of results (one per seed)
    """
    if samplers is None:
        samplers = ["random", "hybrid"]

    all_results = {name: [] for name in samplers}

    for sampler_name in samplers:
        for seed in config.SEEDS:
            results = load_results(config, sampler_name, seed)
            if results is not None:
                all_results[sampler_name].append(results)

    return all_results


# ============================================================================
# SINGLE EXPERIMENT RUNNER
# ============================================================================

def run_single_experiment(
    config: ExperimentConfig,
    sampler_name: str,
    seed: int,
    resume: bool = True
) -> Tuple[List[float], List[float]]:
    """
    Run a single active learning experiment with one sampler and one seed.

    This function handles:
    - Initialization of the pipeline
    - Resuming from checkpoint if available
    - Running AL iterations
    - Saving state and results after each iteration
    - Error handling with state preservation

    Args:
        config: Experiment configuration object
        sampler_name: Name of the sampling strategy ("random" or "hybrid")
        seed: Random seed for reproducibility
        resume: Whether to resume from checkpoint if available

    Returns:
        Tuple of (accuracy_scores, recall_scores) across all iterations
    """
    exp_id = get_experiment_id(sampler_name, seed)
    print(f"\n{'='*80}")
    print(f"[EXPERIMENT] Starting: {exp_id}")
    print(f"{'='*80}")

    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[DEVICE] Using: {device}")

    # Set random seeds
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Check for existing checkpoint to resume from
    start_iteration = 0
    accuracy_scores = []
    recall_scores = []

    if resume:
        latest_iter = get_latest_checkpoint_iteration(config, sampler_name, seed)
        if latest_iter is not None:
            print(f"[RESUME] Found checkpoint at iteration {latest_iter}")

    # Initialize pipeline
    sampler_class = config.SAMPLERS_CONFIG[sampler_name]["class"]
    sampler_params = config.SAMPLERS_CONFIG[sampler_name]["params"]

    try:
        pipeline = sampler_class(
            device=device,
            iterations=config.ITERATIONS,
            epochs_per_iter=config.EPOCHS_PER_ITER,
            budget_per_iter=config.BUDGET_PER_ITER,
            model_name=config.MODEL_NAME,
            objective_function_name=config.LOSS_FUNCTION,
            optimizer_name=config.OPTIMIZER_NAME,
            lr=config.LEARNING_RATE,
            root_dir=config.DATASET_PATH,
            batch_size=config.BATCH_SIZE,
            test_sample_size=config.TEST_SAMPLE_SIZE,
            seed=seed,
            max_train_size=config.MAX_TRAIN_SIZE,
            initial_train_ratio=config.INITIAL_TRAIN_RATIO,
            **sampler_params
        )
        print(f"[PIPELINE] Initialized {sampler_name} sampler")

        # Resume from checkpoint if available
        if resume and latest_iter is not None:
            accuracy_scores, recall_scores = load_experiment_state(
                config, pipeline, sampler_name, seed, latest_iter
            )
            start_iteration = latest_iter + 1
            print(f"[RESUME] Continuing from iteration {start_iteration}")

    except Exception as e:
        print(f"[ERROR] Failed to initialize pipeline: {e}")
        traceback.print_exc()
        raise

    # Run active learning iterations
    try:
        for iteration in range(start_iteration, config.ITERATIONS):
            print(f"\n[ITERATION] {iteration + 1}/{config.ITERATIONS}")
            print(f"[STATUS] Train: {len(pipeline.train_indices)}, Pool: {len(pipeline.pool_indices)}, Test: {len(pipeline.test_indices)}")

            # Check if pool is exhausted
            if len(pipeline.pool_indices) < config.BUDGET_PER_ITER:
                print("[WARNING] Pool exhausted, stopping early")
                save_results(config, sampler_name, seed, accuracy_scores, recall_scores, iteration, "completed")
                break

            # Train model
            print(f"[TRAIN] Training for {config.EPOCHS_PER_ITER} epochs...")
            pipeline._train_model()

            # Evaluate model
            print("[EVAL] Evaluating model...")
            accuracy, recall = pipeline._evaluate_model()
            accuracy_scores.append(accuracy)
            recall_scores.append(recall)

            print(f"[METRICS] Accuracy: {accuracy:.2f}%, Recall: {recall:.2f}%")

            # Save state and results after each iteration
            save_experiment_state(
                config, pipeline, sampler_name, seed, iteration,
                accuracy_scores, recall_scores
            )
            save_results(
                config, sampler_name, seed, accuracy_scores, recall_scores,
                iteration, "in_progress"
            )

            # Sample new points for next iteration (if not last iteration)
            if iteration < config.ITERATIONS - 1:
                print("[SAMPLE] Selecting new samples...")
                new_indices = pipeline._sampling()
                pipeline._update_train_indices(new_indices)
                pipeline._update_pool_indices(new_indices)

        # Mark as completed
        save_results(
            config, sampler_name, seed, accuracy_scores, recall_scores,
            config.ITERATIONS - 1, "completed"
        )
        print(f"\n[COMPLETED] {exp_id}")
        print(f"[FINAL] Accuracy: {accuracy_scores[-1]:.2f}%, Recall: {recall_scores[-1]:.2f}%")

    except Exception as e:
        print(f"\n[ERROR] Experiment failed at iteration {iteration}: {e}")
        traceback.print_exc()

        # Save partial results with failed status
        save_results(
            config, sampler_name, seed, accuracy_scores, recall_scores,
            iteration, "failed"
        )
        print(f"[SAVED] Partial results saved despite failure")
        raise

    return accuracy_scores, recall_scores


# ============================================================================
# MAIN PIPELINE ORCHESTRATION
# ============================================================================

def run_all_experiments(config: ExperimentConfig, resume: bool = True, samplers: Optional[List[str]] = None) -> None:
    """
    Run all experiments across specified samplers and seeds.

    This function orchestrates the full experimental pipeline:
    - For each seed:
        - Run experiments for each specified sampler
    - Handle failures gracefully and continue with remaining experiments

    Args:
        config: Experiment configuration object
        resume: Whether to resume from checkpoints if available
        samplers: List of sampler names to run. If None, runs all available samplers.
    """
    if samplers is None:
        samplers = ["random", "hybrid"]

    print(f"\n{'#'*80}")
    print(f"# ACTIVE LEARNING COMPARISON EXPERIMENTS")
    print(f"#")
    print(f"# Samplers: {samplers}")
    print(f"# Seeds: {config.SEEDS}")
    print(f"# Iterations per experiment: {config.ITERATIONS}")
    print(f"# Total experiments: {len(samplers) * len(config.SEEDS)}")
    print(f"{'#'*80}\n")

    results_summary = {}

    for seed in config.SEEDS:
        print(f"\n{'='*80}")
        print(f"PROCESSING SEED: {seed}")
        print(f"{'='*80}")

        for sampler_name in samplers:
            exp_id = get_experiment_id(sampler_name, seed)

            try:
                # Check if experiment already completed
                existing_results = load_results(config, sampler_name, seed)
                if existing_results and existing_results.get("status") == "completed":
                    print(f"\n[SKIP] {exp_id} already completed")
                    results_summary[exp_id] = "already_completed"
                    continue

                # Run experiment
                accuracy_scores, recall_scores = run_single_experiment(
                    config, sampler_name, seed, resume=resume
                )
                results_summary[exp_id] = "completed"

            except Exception as e:
                print(f"\n[FAILED] {exp_id}: {e}")
                results_summary[exp_id] = f"failed: {str(e)}"
                # Continue with next experiment despite failure
                continue

    # Print summary
    print(f"\n{'#'*80}")
    print(f"# EXPERIMENTS SUMMARY")
    print(f"{'#'*80}")
    for exp_id, status in results_summary.items():
        print(f"{exp_id:30s} -> {status}")
    print(f"{'#'*80}\n")


# ============================================================================
# PLOTTING AND VISUALIZATION
# ============================================================================

def plot_comparison(config: ExperimentConfig, save_path: Optional[str] = None, samplers: Optional[List[str]] = None) -> None:
    """
    Generate comparison plots between samplers.

    This function is independent and can be run separately from the main pipeline.
    It reads results from JSON files and creates:
    - Accuracy comparison across iterations
    - Recall comparison across iterations
    - Mean curves with confidence intervals (shaded regions showing ±1 std)

    Args:
        config: Experiment configuration object
        save_path: Optional path to save plot (default: plots directory)
        samplers: Optional list of sampler names to plot. If None, plots all available results.
    """
    print(f"\n{'='*80}")
    print("[PLOTTING] Generating comparison plots...")
    print(f"{'='*80}")

    # Load all results
    all_results = load_all_results(config, samplers=samplers)

    # Determine which samplers actually have data
    available_samplers = [name for name in all_results.keys() if all_results[name]]

    if not available_samplers:
        print("[ERROR] No results found. Run experiments first.")
        return

    print(f"[PLOTTING] Found results for samplers: {available_samplers}")

    # Prepare data for plotting
    data = {name: {"accuracy": [], "recall": []} for name in available_samplers}

    for sampler_name in available_samplers:
        for result in all_results[sampler_name]:
            if result["status"] == "completed":
                data[sampler_name]["accuracy"].append(result["metrics"]["accuracy_scores"])
                data[sampler_name]["recall"].append(result["metrics"]["recall_scores"])
            else:
                print(f"[WARNING] Skipping incomplete result: {result['experiment_id']} (status: {result['status']})")

    # Check if we have data to plot
    has_data = any(data[name]["accuracy"] for name in available_samplers)
    if not has_data:
        print("[ERROR] No completed experiments found.")
        return

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Create dynamic title based on available samplers
    samplers_str = " vs ".join([s.capitalize() for s in available_samplers])
    fig.suptitle(f'Active Learning Comparison: {samplers_str}\n'
                 f'Model: {config.MODEL_NAME}, Seeds: {config.SEEDS}, Budget: {config.BUDGET_PER_ITER}/iter',
                 fontsize=14, fontweight='bold')

    metrics = ["accuracy", "recall"]
    titles = ["Accuracy (%)", "Recall (%)"]
    colors = {"random": "#FF6B6B", "hybrid": "#4ECDC4", "badge": "#F39C12", "coreset": "#9B59B6"}

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]

        for sampler_name in available_samplers:
            scores = data[sampler_name][metric]

            if not scores:
                continue

            # Convert to numpy array for easier computation
            scores_array = np.array(scores)  # Shape: (n_seeds, n_iterations)

            # Compute mean and std across seeds
            mean_scores = np.mean(scores_array, axis=0)
            std_scores = np.std(scores_array, axis=0)

            # X-axis: iteration numbers
            iterations = np.arange(1, len(mean_scores) + 1)

            # Get color for this sampler (with fallback for unknown samplers)
            sampler_color = colors.get(sampler_name, "#95A5A6")  # Gray fallback

            # Plot mean curve
            label = f"{sampler_name.capitalize()} (n={len(scores)})"
            ax.plot(iterations, mean_scores, 'o-', label=label,
                   color=sampler_color, linewidth=2, markersize=6)

            # Plot confidence interval (±1 std)
            ax.fill_between(iterations,
                           mean_scores - std_scores,
                           mean_scores + std_scores,
                           alpha=0.2, color=sampler_color)

            # Plot individual runs as thin lines
            for run_scores in scores_array:
                ax.plot(iterations, run_scores, '-',
                       color=sampler_color, alpha=0.15, linewidth=1)

        ax.set_xlabel('Active Learning Iteration', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(left=1)

    plt.tight_layout()

    # Save plot
    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(config.PLOTS_DIR, f"comparison_{timestamp}.png")

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[SAVED] Plot saved to: {save_path}")

    # Close the figure to free memory
    plt.close(fig)

    # Print statistics
    print(f"\n{'='*80}")
    print("[STATISTICS]")
    print(f"{'='*80}")

    for sampler_name in available_samplers:
        for metric in ["accuracy", "recall"]:
            scores = data[sampler_name][metric]
            if scores:
                scores_array = np.array(scores)
                final_scores = scores_array[:, -1]  # Last iteration
                print(f"{sampler_name.upper()} {metric.upper()}:")
                print(f"  Final (mean ± std): {np.mean(final_scores):.2f} ± {np.std(final_scores):.2f}")
                print(f"  Final (min-max): {np.min(final_scores):.2f} - {np.max(final_scores):.2f}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """
    Main entry point for the runner.

    Provides two modes:
    1. "full": Run all experiments across seeds and samplers
    2. "plot": Only generate plots from existing results
    """
    parser = argparse.ArgumentParser(
        description="Active Learning Comparison Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full experimental pipeline (both samplers, default config)
  python main_runner.py --mode full

  # Run with custom configuration
  python main_runner.py --mode full --config my_experiment

  # Run only random sampler with custom config
  python main_runner.py --mode full --samplers random --config quick_test

  # Run only hybrid sampler
  python main_runner.py --mode full --samplers hybrid

  # Run both random and hybrid (explicit)
  python main_runner.py --mode full --samplers random hybrid

  # Generate plots from existing results (all available)
  python main_runner.py --mode plot

  # Generate plots only for random sampler
  python main_runner.py --mode plot --samplers random

  # Run without resuming from checkpoints (start fresh)
  python main_runner.py --mode full --no-resume

  # Run with config and no resume
  python main_runner.py --mode full --config my_experiment --no-resume
        """
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "plot"],
        default="full",
        help="Execution mode: 'full' runs experiments, 'plot' only generates plots"
    )
    parser.add_argument(
        "--samplers",
        type=str,
        nargs="+",
        choices=["random", "hybrid"],
        default=None,
        help="Which samplers to run/plot. Default: all available samplers. Examples: --samplers random, --samplers hybrid, --samplers random hybrid"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume from checkpoints (start experiments from scratch)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Configuration name to use (loads from configurations/<name>.json). Use 'default' for built-in defaults."
    )

    args = parser.parse_args()

    # Initialize configuration
    if args.config == "default":
        config = ExperimentConfig()
        print("[CONFIG] Using default configuration")
    else:
        config = ExperimentConfig.from_json(args.config)

    # Setup directories
    setup_directories(config)

    # Execute based on mode
    if args.mode == "full":
        samplers_str = f" ({', '.join(args.samplers)})" if args.samplers else " (all)"
        print(f"[MODE] Running full experimental pipeline{samplers_str}")
        resume = not args.no_resume
        if not resume:
            print("[WARNING] Starting fresh (ignoring existing checkpoints)")
        run_all_experiments(config, resume=resume, samplers=args.samplers)

        # Generate plots after experiments complete
        print("\n[PLOTTING] Generating final comparison plots...")
        plot_comparison(config, samplers=args.samplers)

    elif args.mode == "plot":
        samplers_str = f" ({', '.join(args.samplers)})" if args.samplers else " (all available)"
        print(f"[MODE] Generating plots from existing results{samplers_str}")
        plot_comparison(config, samplers=args.samplers)

    print("\n[DONE] All tasks completed!")


if __name__ == "__main__":
    main()
