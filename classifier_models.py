"""
Classifier Models Module

This module provides pretrained ResNet and DenseNet models for binary classification
of chest X-rays. All models support feature extraction for active learning methods.

The models follow a common architecture:
- Pretrained backbone (ResNet18, ResNet50, or DenseNet121)
- Frozen feature extractor (optional)
- Trainable classifier head for binary classification
- Support for extracting penultimate layer features for BADGE/CoreSet samplers

Classes:
    BaseResnetModel: Abstract base class with common training/evaluation logic
    Resnet18Model: ResNet18-based binary classifier
    Resnet50Model: ResNet50-based binary classifier
    Densenet121Model: DenseNet121-based binary classifier

All models support:
- Binary classification with BCE loss
- Feature extraction for active learning
- Classifier head reinitialization
- Frozen/unfrozen backbone training

Example:
    >>> model = Resnet18Model(optimizer='Adam', loss_function='BCEWithLogitsLoss')
    >>> model.train_model(device, train_loader, epochs=10)
    >>> accuracy, recall = model.evaluate(device, test_loader)
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision import models
from abc import ABC, abstractmethod
from tqdm import tqdm
from torchvision.models import ResNet50_Weights, ResNet18_Weights, DenseNet121_Weights

class BaseResnetModel(nn.Module, ABC):
    """
    Abstract base class for pretrained CNN models with binary classification head.

    This class provides common functionality for:
    - Model initialization with pretrained weights
    - Training loop with progress bars
    - Evaluation with accuracy and recall metrics
    - Feature extraction for active learning
    - Classifier head reinitialization

    The base class handles training, evaluation, and feature extraction while
    subclasses implement model-specific methods for accessing layers.

    Args:
        optimizer (str): Optimizer name ('Adam' or 'SGD')
        loss_function (str): Loss function name ('BCEWithLogitsLoss' or 'CrossEntropyLoss')
        lr (float): Learning rate (default: 0.01)
        freeze (bool): Whether to freeze backbone and only train classifier (default: True)
        pretrained (bool): Whether to use ImageNet pretrained weights (default: True)

    Attributes:
        model: The underlying PyTorch model
        optimizer: PyTorch optimizer
        loss_function: PyTorch loss function

    Note:
        Subclasses must implement:
        - _load_model(): Load and modify the model architecture
        - get_classifier_parameters(): Return classifier parameters
        - get_classifier_module(): Return the classifier head module
        - update_classifier(): Update the classifier head
        - get_penultimate_layer_embeddings(): Extract features before classifier
    """
    #NOTE: All models should return logits, not probabilities!!!
    def __init__(self, optimizer: str = 'Adam', loss_function: str = 'BCEWithLogitsLoss',lr = 1e-2, freeze=True, pretrained=True):
        super(BaseResnetModel, self).__init__()
        self.pretrained = pretrained
        self.model = self._load_model()
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

            classifier_params = self.get_classifier_parameters()
            for param in classifier_params:
                param.requires_grad = True
                
        self._optimizer_name = optimizer
        self._lr = lr
        self._build_optimizer()
        

        if loss_function == 'BCEWithLogitsLoss':
            self.loss_function = torch.nn.BCEWithLogitsLoss()
        elif loss_function == 'CrossEntropyLoss':
            self.loss_function = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Invalid loss function: {loss_function}")
        
    def _build_optimizer(self):
        """
        Build optimizer for trainable parameters only.

        Creates optimizer instance (Adam or SGD) and assigns it to only the parameters
        with requires_grad=True. This is called during __init__ and after classifier
        head reinitialization.

        Raises:
            ValueError: If optimizer name is not 'Adam' or 'SGD'
        """
        trainable = [p for p in self.parameters() if p.requires_grad]
        if self._optimizer_name == 'Adam':
            self.optimizer = torch.optim.Adam(trainable, lr=self._lr)
        elif self._optimizer_name == 'SGD':
            self.optimizer = torch.optim.SGD(trainable, lr=self._lr, momentum=0.9)
        else:
            raise ValueError(f"Invalid optimizer: {self._optimizer_name}")

    def train_model(self, device, dataloader, epochs=3):
        """
        Train the model for a specified number of epochs.

        Args:
            device (torch.device): Device to train on (CPU, CUDA, or MPS)
            dataloader (DataLoader): Training data loader
            epochs (int): Number of epochs to train (default: 3)

        Side Effects:
            - Updates model weights in-place
            - Prints loss after each epoch
        """
        self.model.to(device)
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0
            for images, labels, _ in tqdm(dataloader):
                images = images.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                self.optimizer.zero_grad()
                outputs = self.model(images)

                # temp
                #with torch.no_grad():
                #    print("Output stats:", outputs.min().item(), outputs.max().item())
                # end temp

                loss = self.loss_function(outputs, labels)
                loss.backward()

                # temp
                #for name, param in model.named_parameters():
                #    if param.requires_grad and param.grad is not None:
                #        print(f"{name}: grad norm = {param.grad.norm().item()}")
                # end temp 
                
                self.optimizer.step()

                total_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

    def evaluate(self, device, dataloader):
        """
        Evaluate model performance on a test set.

        Computes accuracy and recall metrics for binary classification using
        a 0.5 threshold on sigmoid probabilities.

        Args:
            device (torch.device): Device to run evaluation on (CPU, CUDA, or MPS)
            dataloader (DataLoader): Test data loader

        Returns:
            tuple: (accuracy, recall) both as percentages
                - accuracy (float): Overall classification accuracy (%)
                - recall (float): Recall for positive class (%)

        Side Effects:
            Prints accuracy and recall to console
        """
        self.model.to(device)
        self.model.eval()
        correct = 0
        total = 0
        true_positives = 0
        actual_positives = 0
        with torch.no_grad():
            for images, labels, _ in tqdm(dataloader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = self.model(images)
                preds = torch.sigmoid(outputs).squeeze() > 0.5
                correct += (preds.int() == labels).sum().item()
                total += labels.size(0)
                true_positives += ((preds.int() == 1) & (labels == 1)).sum().item()
                actual_positives += (labels == 1).sum().item()
        accuracy = correct / total * 100
        recall = (true_positives / actual_positives * 100) if actual_positives > 0 else 0
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Recall: {recall:.2f}%")

        return accuracy, recall

    def reset_classifier_head(self, num_classes: int = None):
        """
        Reinitialize the classification head with fresh random weights.

        This is called at the start of each active learning iteration to reset
        the classifier for training on the newly expanded labeled set.

        Args:
            num_classes (int, optional): Number of output classes (default: 1 for binary)

        Returns:
            nn.Linear: The new classifier head module

        Side Effects:
            - Replaces the classifier head in-place
            - Rebuilds the optimizer to track new parameters
            - Prints confirmation message

        Note:
            Uses Kaiming normal initialization for weights and zeros for bias
        """
        num_classes = 1 if num_classes is None else num_classes
        head = self.get_classifier_module()
        if not hasattr(head, "in_features"):
            raise ValueError("Classifier head does not expose in_features")
        in_f = head.in_features

        new_head = nn.Linear(in_f, num_classes)
        nn.init.kaiming_normal_(new_head.weight, nonlinearity="linear")
        if new_head.bias is not None:
            nn.init.zeros_(new_head.bias)

        self.update_classifier(new_head)

        # ensure the new head is trainable
        for p in self.get_classifier_parameters():
            p.requires_grad = True

        # CRITICAL: rebuild optimizer so it sees the new parameters
        self._build_optimizer()

        print("[CLASSIFIER] Reinitialized the final classifier head and rebuilt optimizer.")
        return new_head

        
    
    def _apply_classifier(self, z: torch.Tensor) -> torch.Tensor:
        """
        Push penultimate features z through the final classifier head to get logits.
        - Accepts z of shape (B, d) or (B, d, 1, 1) and flattens if needed.
        - Returns raw logits: (B, K) for multiclass or (B, 1) for binary.
        """
        if z.dim() > 2:
            z = torch.flatten(z, 1)  # (B, d)
        head = self.get_classifier_module()  # subclass must return the final head (e.g., .fc or .classifier)
        # DDP/DP-safe: (get_classifier_module should already unwrap; this is just extra safety)
        if hasattr(head, "module"):
            head = head.module
        logits = head(z)  # (B, K) or (B, 1)
        # Ensure 2D shape (rare models can return (B,))
        if logits.dim() == 1:
            logits = logits.unsqueeze(1)

        return logits

    @abstractmethod
    def _load_model(self):
        """Return the backbone with the final FC replaced."""
        pass
    
    def forward(self, x, return_features: bool=False, only_features: bool=False):
        """
        Forward pass with optional feature extraction.

        This flexible forward method supports three modes of operation:
        1. Standard: Return logits only
        2. With features: Return both logits and penultimate features
        3. Features only: Return penultimate features without classification

        Args:
            x (torch.Tensor): Input images, shape (B, C, H, W)
            return_features (bool): If True, return (logits, features) tuple
            only_features (bool): If True, return features only (for active learning)

        Returns:
            torch.Tensor or tuple:
                - If only_features=True: Features tensor (B, d)
                - If return_features=True: Tuple of (logits, features)
                - Otherwise: Logits tensor (B, 1) or (B, K)

        Note:
            - logits are raw (NOT probabilities - use sigmoid/softmax externally)
            - Features are from the penultimate layer before classifier
            - Used by BADGE and CoreSet active learning samplers
        """
        z = self.get_penultimate_layer_embeddings(x)  # (B, d)
        if only_features:
            return z
        logits = self._apply_classifier(z)            # (B, K) or (B, 1)
        return (logits, z) if return_features else logits

    def get_classifier_parameters(self):
        raise NotImplementedError("Subclasses must implement get_classifier_parameters method")

    def get_classifier_module(self):
        raise NotImplementedError("Subclasses must implement get_classifier_parameters method")

    def update_classifier(self, new_features):
        raise NotImplementedError("Subclasses must implement update_classifier method")

    @abstractmethod
    def get_penultimate_layer_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts the output of the penultimate layer (before the final classifier).
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement get_classifier_parameters method")


class Resnet50Model(BaseResnetModel):
    """
    ResNet50-based binary classifier for chest X-ray images.

    Uses ImageNet-pretrained ResNet50 backbone with a single-output
    linear classifier head for binary classification.

    Architecture:
        - Backbone: ResNet50 (2048-dimensional features)
        - Classifier: Linear(2048, 1) for binary classification
        - Feature dim: 2048

    Inherits all functionality from BaseResnetModel including training,
    evaluation, and feature extraction for active learning.

    Example:
        >>> model = Resnet50Model(optimizer='Adam', lr=0.001)
        >>> model.train_model(device, train_loader, epochs=10)
        >>> logits, features = model.forward(images, return_features=True)
    """
    def _load_model(self):
        if self.pretrained:
            print("Loading pretrained ResNet50 model...")
            resnet50_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            print("Loading ResNet50 model without pretrained weights...")
            resnet50_model = models.resnet50(weights=None)

        resnet50_model.fc = nn.Linear(resnet50_model.fc.in_features, 1)
        return resnet50_model

    def get_classifier_parameters(self):
        return self.model.fc.parameters()
    
    def get_classifier_module(self):
        return self.model.fc
    
    def update_classifier(self, new_head: nn.Module):
        self.model.fc = new_head

    def get_penultimate_layer_embeddings(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x



class Resnet18Model(BaseResnetModel):
    """
    ResNet18-based binary classifier for chest X-ray images.

    Uses ImageNet-pretrained ResNet18 backbone with a single-output
    linear classifier head for binary classification. Lighter and faster
    than ResNet50 while maintaining good performance.

    Architecture:
        - Backbone: ResNet18 (512-dimensional features)
        - Classifier: Linear(512, 1) for binary classification
        - Feature dim: 512

    Inherits all functionality from BaseResnetModel including training,
    evaluation, and feature extraction for active learning.

    Example:
        >>> model = Resnet18Model(optimizer='Adam', lr=0.001)
        >>> model.train_model(device, train_loader, epochs=10)
        >>> features = model.forward(images, only_features=True)
    """
    def _load_model(self):
        if self.pretrained:
            print("Loading pretrained ResNet18 model...")
            resnet18_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            print("Loading ResNet18 model without pretrained weights...")
            resnet18_model = models.resnet18(weights=None)

        resnet18_model.fc = nn.Linear(resnet18_model.fc.in_features, 1)
        return resnet18_model

    def get_classifier_parameters(self):
        return self.model.fc.parameters()

    def get_classifier_module(self):
        return self.model.fc

    def update_classifier(self, new_head: nn.Module):
        self.model.fc = new_head

    def get_penultimate_layer_embeddings(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x



class Densenet121Model(BaseResnetModel):
    """
    DenseNet121-based binary classifier for chest X-ray images.

    Uses ImageNet-pretrained DenseNet121 backbone with a single-output
    linear classifier head for binary classification. DenseNet uses dense
    connections and can be more parameter-efficient than ResNets.

    Architecture:
        - Backbone: DenseNet121 (1024-dimensional features)
        - Classifier: Linear(1024, out_size) for binary classification
        - Feature dim: 1024

    Args:
        out_size (int): Number of output classes (default: 1 for binary)
        optimizer (str): Optimizer name ('Adam' or 'SGD')
        loss_function (str): Loss function name
        lr (float): Learning rate
        freeze (bool): Whether to freeze backbone
        pretrained (bool): Whether to use ImageNet pretrained weights

    Inherits all functionality from BaseResnetModel including training,
    evaluation, and feature extraction for active learning.

    Example:
        >>> model = Densenet121Model(optimizer='Adam', lr=0.001)
        >>> model.train_model(device, train_loader, epochs=10)
        >>> accuracy, recall = model.evaluate(device, test_loader)
    """
    def __init__(self, out_size=1, optimizer='Adam', loss_function='BCEWithLogitsLoss',lr=1e-2, freeze=True, pretrained=True):
        self.out_size = out_size
        super().__init__(optimizer, loss_function, lr, freeze, pretrained)

    def _load_model(self):
        if self.pretrained:
            print("Loading pretrained DenseNet121 model...")
            densenet121_model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
        else:
            print("Loading DenseNet121 model without pretrained weights...")
            densenet121_model = models.densenet121(weights=None)

        num_ftrs = densenet121_model.classifier.in_features
        densenet121_model.classifier = nn.Linear(num_ftrs, self.out_size)
        return densenet121_model

    def get_classifier_parameters(self):
        return self.model.classifier.parameters()

    def get_classifier_module(self):
        return self.model.classifier

    def update_classifier(self, new_head: nn.Module):
        self.model.classifier = new_head

    def get_penultimate_layer_embeddings(self, x):
        features = self.model.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(x.size(0), -1)
        return out  # shape [B, 1024]
