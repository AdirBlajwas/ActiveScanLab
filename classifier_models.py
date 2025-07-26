import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from abc import ABC, abstractmethod
from tqdm import tqdm

class BaseResnetModel(nn.Module, ABC):
    #NOTE: All models should return logits, not probabilities!!!
    def __init__(self, optimizer: str = 'Adam', loss_function: str = 'BCEWithLogitsLoss', freeze=True, pretrained=True,):
        super(BaseResnetModel, self).__init__()
        self.pretrained = pretrained
        self.model = self._load_model()

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

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

            classifier_params = self.get_classifier_parameters()
            for param in classifier_params:
                param.requires_grad = True
        
    
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
        with torch.no_grad():
            for images, labels, _ in dataloader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = self.model(images)
                preds = torch.sigmoid(outputs).squeeze() > 0.5
                correct += (preds.int() == labels).sum().item()
                total += labels.size(0)
        print(f"Accuracy: {correct / total * 100:.2f}%")


                
    @abstractmethod
    def _load_model(self):
        """Return the backbone with the final FC replaced."""
        pass
    
    def forward(self, x):
        return self.model(x)

    def gradient_embedding(self, x):
        """
        Computes gradient embeddings for the BADGE sampling method.
        """
        x = x.clone().detach().requires_grad_(True)
        logits = self.forward(x)

        pseudo_labels = (logits > 0).float()
        loss = F.binary_cross_entropy_with_logits(logits, pseudo_labels, reduction='sum')
        loss.backward()

        embeddings = x.grad.view(x.size(0), -1).cpu().numpy()

        x.grad.zero_()

        return embeddings

    def get_classifier_parameters(self):
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


class Densenet121Model(BaseResnetModel):
    def __init__(self, out_size=1, optimizer='Adam', loss_function='BCEWithLogitsLoss', freeze=True, pretrained=True):
        self.out_size = out_size
        super().__init__(optimizer, loss_function, freeze, pretrained)
        self.model = self._load_model()

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