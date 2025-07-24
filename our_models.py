import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from abc import ABC, abstractmethod

class BaseResNetModel(nn.Module, ABC):
    #NOTE: All models should return logits, not probabilities!!!
    def __init__(self, freeze=True, pretrained=True):
        super(BaseResNetModel, self).__init__()
        self.pretrained = pretrained
        self.model = self._load_model()

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

            classifier_params = self.get_classifier_parameters()
            for param in classifier_params:
                param.requires_grad = True
                
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


class ResNet50Model(BaseResNetModel):
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


class ResNet18Model(BaseResNetModel):
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


class DenseNet121Model(BaseResNetModel):
    def __init__(self, out_size=1, freeze=True, pretrained=True):
        self.out_size = out_size
        super(DenseNet121Model, self).__init__(freeze=freeze, pretrained=pretrained)

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