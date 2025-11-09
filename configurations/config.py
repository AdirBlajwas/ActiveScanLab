"""
Configuration Module for Active Learning Experiments

This module provides the ExperimentConfig class for managing experiment hyperparameters.
Default values are loaded from default_config.json and can be overridden by custom
JSON configuration files.

Usage:
    # Use default configuration
    config = ExperimentConfig()

    # Load custom configuration
    config = ExperimentConfig.from_json("run1")

    # Access configuration values
    print(config.MODEL_NAME)
    print(config.BATCH_SIZE)
"""

import os
import json
from typing import Optional

# Import sampler classes for SAMPLERS_CONFIG
from active_learning_models import (
    RandomSamplingActiveLearning,
    HybridBADGECoreSetActiveLearning,
)


class ExperimentConfig:
    """
    Centralized configuration for all experiment hyperparameters.

    Loads default values from default_config.json and supports overriding
    specific values via custom JSON configuration files.

    Attributes:
        CONFIG_NAME (str): Name of the configuration being used
        DATASET_PATH (str): Path to the NIH chest X-ray dataset
        BATCH_SIZE (int): Batch size for training and evaluation
        EPOCHS_PER_ITER (int): Number of training epochs per AL iteration
        MODEL_NAME (str): Model architecture ("resnet18", "resnet50", "densenet121")
        OPTIMIZER_NAME (str): Optimizer ("Adam" or "SGD")
        LOSS_FUNCTION (str): Loss function for binary classification
        LEARNING_RATE (float): Learning rate for optimizer
        ITERATIONS (int): Number of AL iterations to run
        BUDGET_PER_ITER (int): Number of samples to label per iteration
        TEST_SAMPLE_SIZE (int): Number of test samples to evaluate on
        MAX_TRAIN_SIZE (int): Maximum training set size
        INITIAL_TRAIN_RATIO (float): Fraction of training pool for initial labeled set
        SEEDS (list): List of random seeds for reproducibility
        SAMPLERS_CONFIG (dict): Configuration for each sampler with class references
        RESULTS_DIR (str): Directory to save results
        CHECKPOINT_DIR (str): Directory to save checkpoints
        PLOTS_DIR (str): Directory to save plots
        CONFIGS_DIR (str): Directory containing configuration files

    Example:
        >>> config = ExperimentConfig()
        >>> print(config.MODEL_NAME)
        'resnet18'

        >>> config = ExperimentConfig.from_json("run1")
        >>> print(config.ITERATIONS)
        10
    """

    def __init__(self):
        """
        Initialize configuration with default values from default_config.json.
        """
        # Get the directory where this config.py file is located
        config_dir = os.path.dirname(os.path.abspath(__file__))
        default_config_path = os.path.join(config_dir, "default_config.json")

        # Load default configuration
        if not os.path.exists(default_config_path):
            raise FileNotFoundError(
                f"Default configuration file not found: {default_config_path}\n"
                f"Please ensure default_config.json exists in the configurations/ directory"
            )

        with open(default_config_path, "r") as f:
            default_config = json.load(f)

        # Set all configuration values from default_config.json
        for key, value in default_config.items():
            setattr(self, key, value)

        # Add sampler class references to SAMPLERS_CONFIG
        # (These can't be stored in JSON, so we add them programmatically)
        self.SAMPLERS_CONFIG["random"]["class"] = RandomSamplingActiveLearning
        self.SAMPLERS_CONFIG["hybrid"]["class"] = HybridBADGECoreSetActiveLearning

    @classmethod
    def from_json(cls, config_name: str) -> 'ExperimentConfig':
        """
        Load configuration from a custom JSON file, overriding default values.

        Args:
            config_name: Name of the configuration file (without .json extension)
                        File should be at: configurations/<config_name>.json

        Returns:
            ExperimentConfig instance with values loaded from custom JSON

        Example JSON structure (configurations/my_experiment.json):
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

        Example:
            >>> config = ExperimentConfig.from_json("run1")
            >>> print(config.ITERATIONS)
            10
        """
        # Start with default configuration
        config = cls()
        config.CONFIG_NAME = config_name

        # Load custom JSON file
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
            Dictionary with all configuration values (excluding class methods and class references)

        Example:
            >>> config = ExperimentConfig()
            >>> config_dict = config.to_dict()
            >>> print(config_dict["MODEL_NAME"])
            'resnet18'
        """
        config_dict = {}
        for key in dir(self):
            if key.isupper() and not key.startswith("_"):
                value = getattr(self, key)
                # Skip SAMPLERS_CONFIG as it contains class references
                if key != "SAMPLERS_CONFIG":
                    config_dict[key] = value
        return config_dict