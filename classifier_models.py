import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision import models
from abc import ABC, abstractmethod
from tqdm import tqdm

class BaseResnetModel(nn.Module, ABC):
    #NOTE: All models should return logits, not probabilities!!!
    def __init__(self, optimizer: str = 'Adam', loss_function: str = 'BCEWithLogitsLoss', freeze=True, pretrained=True):
        super(BaseResnetModel, self).__init__()
        self.pretrained = pretrained
        self.model = self._load_model()
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

            classifier_params = self.get_classifier_parameters()
            for param in classifier_params:
                param.requires_grad = True

        if optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        elif optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
        else:
            raise ValueError(f"Invalid optimizer: {optimizer}")

        if loss_function == 'BCEWithLogitsLoss':
            self.loss_function = torch.nn.BCEWithLogitsLoss()
        elif loss_function == 'CrossEntropyLoss':
            self.loss_function = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Invalid loss function: {loss_function}")
        
    
    def train_model(self, device, dataloader, epochs=3):
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
        z = self.get_penultimate_layer_embeddings(x)  # (B, d)
        if only_features:
            return z
        logits = self._apply_classifier(z)            # (B, K) or (B, 1)
        return (logits, z) if return_features else logits

    def get_classifier_parameters(self):
        raise NotImplementedError("Subclasses must implement get_classifier_parameters method")
    
    def get_classifier_module(self):
        raise NotImplementedError("Subclasses must implement get_classifier_parameters method")
    
    @abstractmethod
    def get_penultimate_layer_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts the output of the penultimate layer (before the final classifier).
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement get_classifier_parameters method")


class Resnet50Model(BaseResnetModel):
    def _load_model(self):
        if self.pretrained:
            print("Loading pretrained ResNet50 model...")
            resnet50_model = models.resnet50(weights='DEFAULT')
        else:
            print("Loading ResNet50 model without pretrained weights...")
            resnet50_model = models.resnet50(weights=None)

        resnet50_model.fc = nn.Linear(resnet50_model.fc.in_features, 1)
        return resnet50_model

    def get_classifier_parameters(self):
        return self.model.fc.parameters()
    
    def get_classifier_module(self):
        return self.model.fc


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
    def _load_model(self):
        if self.pretrained:
            print("Loading pretrained ResNet18 model...")
            resnet18_model = models.resnet18(weights='DEFAULT')
        else:
            print("Loading ResNet18 model without pretrained weights...")
            resnet18_model = models.resnet18(weights=None)

        resnet18_model.fc = nn.Linear(resnet18_model.fc.in_features, 1)
        return resnet18_model

    def get_classifier_parameters(self):
        return self.model.fc.parameters()
    
    def get_classifier_module(self):
        return self.model.fc

    
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
    def __init__(self, out_size=1, optimizer='Adam', loss_function='BCEWithLogitsLoss', freeze=True, pretrained=True):
        self.out_size = out_size
        super().__init__(optimizer, loss_function, freeze, pretrained)

    def _load_model(self):
        if self.pretrained:
            print("Loading pretrained DenseNet121 model...")
            densenet121_model = models.densenet121(weights='DEFAULT')
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

    
    def get_penultimate_layer_embeddings(self, x):
        features = self.model.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(x.size(0), -1)
        return out  # shape [B, 1024]
