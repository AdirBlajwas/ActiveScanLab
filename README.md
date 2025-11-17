# ActiveScanLab: Active Learning for Chest X-ray Classification

A research framework for comparing active learning strategies (Random vs Hybrid BADGE+CoreSet) on medical image classification using the NIH Chest X-ray dataset.

---

## 🚀 Quick Start

### 1. Setup Environment

#### 1.1 Create Virtual Environment

Clone the repository and set up the Python environment:

```bash
# Clone the repository
git clone https://github.com/AdirBlajwas/ActiveScanLab.git
cd ActiveScanLab

# Run the setup script
chmod +x setup_venv.sh
./setup_venv.sh
```

This will:
- Create a Python virtual environment in `venv/`
- Install all required dependencies (PyTorch, NumPy, Pandas, etc.)

#### 1.2 Download and Prepare Dataset

1. **Download the dataset** from Google Drive:
   ```
   https://drive.google.com/file/d/1QvS_H5ucGGTcWwgC707uN7cVQmB-9P9l/view?usp=sharing
   ```

2. **Unzip the dataset** into the main project directory:
   ```bash
   # After downloading, unzip the file
   unzip nih_chest_xrays_light.zip

   # Ensure the directory structure looks like this:
   # ActiveScanLab/
   # ├── nih_chest_xrays_light/
   # │   ├── images_001_lighter/
   # │   ├── images_002_lighter/
   # │   ├── ...
   # │   ├── Data_Entry_2017.csv
   # │   ├── train_val_list.txt
   # │   └── test_list.txt
   # ├── main.py
   # └── ...
   ```

3. **Verify the dataset** is properly placed:
   ```bash
   ls nih_chest_xrays_light/
   # Should show: images_001_lighter, images_002_lighter, ..., Data_Entry_2017.csv
   ```

---

## 🔬 Reproducing Project Results

### 2.1 Run Experiments with Configurations

Activate the virtual environment and run experiments with the three main configurations:

```bash
# Activate virtual environment
source venv/bin/activate

# Run Experiment 1 (80 epochs, 10 iterations, 10% initial labeled)
python main.py --mode full --config run1

# Run Experiment 2 (100 epochs, 12 iterations, 10% initial labeled)
python main.py --mode full --config run2

```

Each command will:
- Run both **Random** and **Hybrid** active learning samplers
- Train across multiple seeds for statistical robustness
- Save results to `results/` directory
- Save checkpoints to `checkpoints_main/` for automatic resumption
- Generate comparison plots in `plots/` directory

**Note:** These experiments can take several hours to complete. The pipeline automatically saves checkpoints after each iteration, so you can safely interrupt and resume later by running the same command.

If you want just to reproduce the plots from existing results without rerunning experiments, use:

```bash
# Activate virtual environment
source venv/bin/activate
python main.py --mode plot --config run1
python main.py --mode plot --config run2
```

### 2.2 Advanced Usage

For detailed information about all available options, configurations, and workflows, see:

📖 **[MAIN_RUNNING_GUIDE.md](MAIN_RUNNING_GUIDE.md)**

This guide covers:
- Running specific samplers only (`--samplers random` or `--samplers hybrid`)
- Creating custom configurations
- Plotting existing results (`--mode plot`)
- Resumption control (`--no-resume`)
- Complete parameter reference

---

## 📁 Repository Structure

### Key Python Files

#### `main.py`
Main experimental pipeline orchestrator. Handles:
- Configuration management (JSON-based configs)
- Multi-seed experiment execution
- Automatic checkpointing and resume
- Results aggregation and plotting
- Command-line interface

#### `active_learning_models.py`
Implementation of active learning sampling strategies:
- **`ActiveLearningPipeline`**: Base class with training loop and state management
- **`RandomSamplingActiveLearning`**: Random baseline sampler
- **`BADGESamplingActiveLearning`**: Gradient-based uncertainty sampling (BADGE method)
- **`CoreSetSamplingActiveLearning`**: Diversity-based sampling (greedy k-center)
- **`HybridBADGECoreSetActiveLearning`**: Combined BADGE + CoreSet approach

#### `classifier_models.py`
Deep learning model architectures for binary classification:
- **`BaseResnetModel`**: Abstract base class with training/evaluation logic
- **`Resnet18Model`**: ResNet18 backbone (512-dim features)
- **`Resnet50Model`**: ResNet50 backbone (2048-dim features)
- **`Densenet121Model`**: DenseNet121 backbone (1024-dim features)

All models support feature extraction for active learning samplers.

#### `dataset.py`
NIH Chest X-ray dataset handling:
- **`ChestXrayDataset`**: Main dataset class with flexible train/test splitting
- Automatic image discovery across multiple folders
- Binary classification: 0 = No Finding, 1 = Finding

### Directories

#### `configurations/`
JSON configuration files for experiments:
- **`run1.json`**: 80 epochs, 10 iterations, 1000 budget/iter
- **`run2.json`**: 100 epochs, 12 iterations, 1000 budget/iter

Each config specifies hyperparameters like epochs, iterations, budget, initial labeled ratio, and sampler parameters.

#### `results/`
Experiment results in JSON format:
- Accuracy and recall scores per iteration
- Full experiment configuration
- Execution status and timestamps
- Naming format: `{sampler}_seed{seed}_config_{config}_results.json`

#### `checkpoints_main/`
Model checkpoints for automatic resumption:
- Saved after each active learning iteration
- Contains model weights, optimizer state, and train/pool indices
- Enables seamless experiment resumption after interruption

#### `plots/`
Generated visualization plots:
- Comparison plots with mean ± std deviation
- Individual seed trajectories
- Saved as PNG files with timestamps

#### `tests/`
Testing and debugging utilities:
- Sampler-specific test scripts
- Data preprocessing notebooks
- Experimental validation code

---
### Dependencies

Core libraries (installed automatically via `setup_venv.sh`):
- PyTorch >= 2.0.0
- TorchVision >= 0.15.0
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.5.0
- tqdm >= 4.62.0

See `requirements.txt` for complete list.

---

## 📊 Expected Outputs

After running experiments, you should have:

1. **Results** in `results/` directory:
   - JSON files with accuracy/recall trajectories
   - One file per sampler-seed-config combination

2. **Checkpoints** in `checkpoints_main/` directory:
   - Model states for each iteration
   - Enables resumption if interrupted

3. **Plots** in `plots/` directory:
   - Comparison visualizations
   - Mean performance ± standard deviation across seeds

---

## 🐛 Troubleshooting

**Dataset not found?**
- Ensure `nih_chest_xrays_light/` is in the project root directory
- Check that `Data_Entry_2017.csv` exists inside

**CUDA out of memory?**
- Reduce `BATCH_SIZE` in configuration files
- Lower `badge_subsample` and `coreset_subsample` parameters
- Use `badge_fp16: true` for memory efficiency

**Virtual environment issues?**
- Make sure you activated: `source venv/bin/activate`
- Reinstall: `rm -rf venv && ./setup_venv.sh`

**Experiments interrupted?**
- Simply rerun the same command - it will auto-resume from the last checkpoint
- To start fresh: add `--no-resume` flag


---

## 📄 License

**[MIT License](LICENSE)**

---

