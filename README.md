# ActiveScanLab

Active Learning for Medical Image Classification using PyTorch.

## Setup

### 1. Create Virtual Environment

Run the setup script to create and configure the virtual environment:

```bash
chmod +x setup_venv.sh
./setup_venv.sh
```

### 2. Activate Virtual Environment

```bash
source venv/bin/activate
```

Or use the activation script:

```bash
chmod +x activate_venv.sh
./activate_venv.sh
```

### 3. Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torchvision; print(f'TorchVision version: {torchvision.__version__}')"
```

## Usage

### Running the Main Pipeline

```bash
python run.py --mode pipeline --model resnet50 --epochs 3
```

### Running Individual Components

```bash
# Preprocessing
python run.py --mode preprocessing

# Training
python run.py --mode training --model resnet18 --epochs 5

# Evaluation
python run.py --mode evaluation --model resnet50
```

### Running Jupyter Notebooks

With the virtual environment activated:

```bash
jupyter notebook
```

## Project Structure

- `active_learning_models.py` - Active learning pipeline implementations
- `classifier_models.py` - Neural network models (ResNet, DenseNet)
- `custom_dataset.py` - Dataset loading and preprocessing
- `data_loader.py` - Data loading utilities
- `preprocessing.ipynb` - Data preprocessing notebook
- `run_pipelines_clean.ipynb` - Main pipeline execution notebook
- `run.py` - Command-line interface for running experiments

## VS Code Configuration

The project includes VS Code configuration files that:
- Set the Python interpreter to the virtual environment
- Configure the workspace root
- Enable automatic environment activation
- Set up debugging configurations

## Dependencies

- PyTorch >= 2.0.0
- TorchVision >= 0.15.0
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- Pillow >= 8.3.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.5.0
- tqdm >= 4.62.0
- Jupyter >= 1.0.0

## License

MIT License
