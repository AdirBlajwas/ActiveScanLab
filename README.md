# Active Learning Comparison Framework

Compare active learning sampling strategies (Random vs Hybrid BADGE+CoreSet) on chest X-ray classification.

---

## Quick Start

```bash
# Run full experiment (default settings)
python main_runner.py --mode full

# View results
python main_runner.py --mode plot
```

---

## Command-Line Options

### 🎯 Execution Mode

```bash
--mode [full|plot]
```

- `full` - Run experiments + generate plots
- `plot` - Only generate plots from saved results

### 🔬 Select Samplers

```bash
--samplers [random] [hybrid]
```

**Examples:**
```bash
# Random sampler only
python main_runner.py --mode full --samplers random

# Hybrid sampler only
python main_runner.py --mode full --samplers hybrid

# Both (default if omitted)
python main_runner.py --mode full --samplers random hybrid
```

### ⚙️ Custom Configuration

```bash
--config CONFIG_NAME
```

Load settings from `configurations/CONFIG_NAME.json`

**Examples:**
```bash
# Quick test (3 iterations, fast)
python main_runner.py --mode full --config quick_test

# From log file configurations
python main_runner.py --mode full --config run1
python main_runner.py --mode full --config run2
python main_runner.py --mode full --config run3
```

**Default:** Uses built-in defaults when omitted

### 🔄 Resume Control

```bash
--no-resume
```

Start fresh, ignore checkpoints

**Examples:**
```bash
# Auto-resume from checkpoint (default)
python main_runner.py --mode full --config run1

# Start from scratch
python main_runner.py --mode full --config run1 --no-resume
```

---

## Common Usage Patterns

### Quick Testing

```bash
# Fast test before full run
python main_runner.py --mode full --config quick_test

# If good, run production
python main_runner.py --mode full --config run1
```

### Production Runs

```bash
# Full experiment (both samplers, auto-resume)
python main_runner.py --mode full --config run1

# Hybrid only with run2 settings
python main_runner.py --mode full --samplers hybrid --config run2

# Fresh start without resume
python main_runner.py --mode full --config run3 --no-resume
```

### Plotting

```bash
# Plot all available results
python main_runner.py --mode plot

# Plot specific sampler
python main_runner.py --mode plot --samplers random

# Plot specific configuration
python main_runner.py --mode plot --config run1
```

### Combined Options

```bash
# Hybrid only, run3 config, fresh start
python main_runner.py --mode full --samplers hybrid --config run3 --no-resume

# Random only, quick test
python main_runner.py --mode full --samplers random --config quick_test
```

---

## Configuration Files

### 📁 Location

All configs stored in `configurations/` directory as JSON files.

### 📋 Available Configurations

| Config | Epochs | Iterations | Budget | Initial % | Description |
|--------|--------|------------|--------|-----------|-------------|
| `quick_test` | 1 | 3 | 500 | 10% | Fast testing |
| `run1` | 80 | 10 | 1000 | 10% | Balanced run |
| `run2` | 100 | 12 | 1000 | 10% | More epochs |
| `run3` | 80 | 16 | 2500 | 8% | Large budget |

### ✏️ Create Custom Config

Create `configurations/my_experiment.json`:

```json
{
  "ITERATIONS": 5,
  "EPOCHS_PER_ITER": 50,
  "BUDGET_PER_ITER": 1000,
  "SEEDS": [42],
  "MODEL_NAME": "resnet50",
  "INITIAL_TRAIN_RATIO": 0.1
}
```

Run with:
```bash
python main_runner.py --mode full --config my_experiment
```

### 🔧 Configuration Parameters

**Training:**
- `BATCH_SIZE` (32) - Training batch size
- `EPOCHS_PER_ITER` (2) - Epochs per AL iteration
- `MODEL_NAME` ("resnet18") - `resnet18` | `resnet50` | `densenet121`
- `LEARNING_RATE` (0.001) - Learning rate
- `OPTIMIZER_NAME` ("Adam") - `Adam` | `SGD`

**Active Learning:**
- `ITERATIONS` (10) - Number of AL iterations
- `BUDGET_PER_ITER` (1000) - Samples to label per iteration
- `TEST_SAMPLE_SIZE` (1000) - Test set size (`null` = all)
- `INITIAL_TRAIN_RATIO` (0.1) - Initial labeled set % (0.1 = 10%)
- `SEEDS` ([42, 7, 1]) - Random seeds list

**Hybrid Sampler (optional):**
```json
"SAMPLERS_CONFIG": {
  "hybrid": {
    "params": {
      "badge_ratio": 0.5,
      "badge_subsample": 20000,
      "badge_fp16": false,
      "coreset_subsample": 20000,
      "l2_norm": true,
      "dist_chunk": 2048
    }
  }
}
```

**Partial Override:** Only specify parameters you want to change. Others use defaults.

---

## Output Files

### 📊 Results (`results/`)

**Naming:**
- Default config: `random_seed42_results.json`
- Custom config: `random_seed42_config_run1_results.json`

**Contains:**
- Accuracy & recall scores per iteration
- Experiment configuration
- Status (completed/in_progress/failed)
- Timestamp

### 💾 Checkpoints (`checkpoints_main/`)

- Auto-saved after each iteration
- Contains: model weights, optimizer state, train/pool indices
- Enables automatic resume after interruption
- Naming includes config name

### 📈 Plots (`plots/`)

- Mean curves ± standard deviation
- Individual run trajectories (faded)
- Timestamp-based filenames

---

## Complete Command Reference

```bash
# ========== FULL EXPERIMENTS ==========

# All defaults (both samplers, default config, auto-resume)
python main_runner.py --mode full

# With custom config
python main_runner.py --mode full --config run1

# Single sampler
python main_runner.py --mode full --samplers random
python main_runner.py --mode full --samplers hybrid

# Single sampler + config
python main_runner.py --mode full --samplers hybrid --config run2

# Fresh start
python main_runner.py --mode full --no-resume
python main_runner.py --mode full --config run1 --no-resume

# ========== PLOTTING ==========

# All results
python main_runner.py --mode plot

# Specific sampler
python main_runner.py --mode plot --samplers random

# Specific config
python main_runner.py --mode plot --config run1

# ========== COMBINED OPTIONS ==========

python main_runner.py --mode full --samplers hybrid --config run3 --no-resume
python main_runner.py --mode full --samplers random hybrid --config run2
```

---

## Workflows

### 🧪 1. Test → Production

```bash
# Step 1: Quick test
python main_runner.py --mode full --config quick_test

# Step 2: If good, full run
python main_runner.py --mode full --config run1
```

### 🔬 2. Sampler Comparison

```bash
# Run both
python main_runner.py --mode full --config run1

# Or run separately
python main_runner.py --mode full --samplers random --config run1
python main_runner.py --mode full --samplers hybrid --config run1

# Compare
python main_runner.py --mode plot --config run1
```

### 🏗️ 3. Interrupted Experiment

```bash
# Experiment was interrupted...

# Just run same command - auto-resumes!
python main_runner.py --mode full --config run2

# Or start over:
python main_runner.py --mode full --config run2 --no-resume
```

### 📊 4. Model Comparison

```bash
# Create configs: resnet18.json, resnet50.json, densenet.json

python main_runner.py --mode full --config resnet18
python main_runner.py --mode full --config resnet50
python main_runner.py --mode full --config densenet

# Compare all
python main_runner.py --mode plot
```

---

## Key Features

✅ **Auto-Checkpointing** - Never lose progress
✅ **Auto-Resume** - Continue from checkpoint by default
✅ **Multi-Seed** - Statistical robustness
✅ **JSON Config** - Flexible, version-controllable
✅ **Partial Override** - Only specify what changes
✅ **Config Tracking** - Config name in all output files
✅ **Independent Plotting** - Regenerate plots anytime
✅ **Sampler Filtering** - Run specific samplers only

---

## Troubleshooting

**Q: Experiments not resuming?**
A: Check `checkpoints_main/` for checkpoint files. Use `--no-resume` to start fresh.

**Q: No plots generated?**
A: Ensure completed experiments exist in `results/` directory.

**Q: Out of memory?**
A: Reduce `BATCH_SIZE`, lower `badge_subsample`/`coreset_subsample`, or use `"badge_fp16": true`

**Q: Config file not found?**
A: Check file exists: `configurations/YOUR_CONFIG.json`

**Q: How to see all options?**
A: Run `python main_runner.py --help`

---

## Project Structure

```
ActiveScanLab/
├── main_runner.py              # Main pipeline ⭐
├── active_learning_models.py   # AL samplers
├── classifier_models.py        # Model architectures
├── custom_dataset.py           # Dataset handling
├── configurations/             # Config files
│   ├── quick_test.json        # Fast testing
│   ├── run1.json              # Experiment 1
│   ├── run2.json              # Experiment 2
│   └── run3.json              # Experiment 3
├── results/                    # JSON results
├── checkpoints_main/           # Model checkpoints
└── plots/                      # Generated plots
```

---

## Setup (First Time Only)

### 1. Create Virtual Environment

```bash
chmod +x setup_venv.sh
./setup_venv.sh
```

### 2. Activate Environment

```bash
source venv/bin/activate
```

### 3. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

---

## Dependencies

- PyTorch >= 2.0.0
- TorchVision >= 0.15.0
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.5.0
- tqdm >= 4.62.0

---

## Need Help?

```bash
python main_runner.py --help
```

Shows complete CLI documentation with examples.
