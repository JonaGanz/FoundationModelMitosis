import torch
import torch.nn as nn
import peft
from .utils import load_model_and_transforms
from transformers import ViTModel
import os

# Define where the LoRA adaptation can be applied for each model
VALID_LORA_MODULES = {
    "hoptimus": ["qkv", "proj", "fc1", "fc2"],
    "virchow2": ["qkv", "proj", "fc1", "fc2"],
    "virchow": ["qkv", "proj", "fc1", "fc2"],
    "ViT_H": ["query", "key", "value", "dense"],
    "uni": ["qkv", "proj", "fc1", "fc2"],
    "gigapath": ["qkv", "proj", "fc1", "fc2"],
    "phikon": ["query", "key", "value", "dense"],
}
# Define which models are ViTs which do not have a classification head
# and require a custom head to be added   
VIT_FOR_CLASSIFICATION = {
    "ViT_H",
    "phikon"
}

class ViTForClassification(nn.Module):
    """
    Vision Transformer (ViT) model for classification tasks.

    Args:
        model_name (str): Name of the ViT model to load.
        num_classes (int, optional): Number of output classes. Defaults to 2.
    """
    def __init__(self, model_name: str, num_classes: int = 2):
        super().__init__()
        self.vit = self.load_backbone(model_name)

        # Replace the classification head (default is Identity)
        num_classes = 1 if num_classes == 2 else num_classes
        
        if self.head_input_units(model_name) is None:
            raise ValueError(f"Unknown model: {model_name}. Please update head_input_units().")
        
        self.head = nn.Linear(self.head_input_units(model_name), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ViT model.

        Args:
            x (torch.Tensor): Input tensor with shape [B, 3, W, H] (image batch).

        Returns:
            torch.Tensor: Logits with shape [B, num_classes].
        """
        outputs = self.vit(x)  # Get the full output of the transformer

        # Extract the [CLS] token (first token in sequence)
        cls_token = outputs.last_hidden_state[:, 0, :]  # Shape: (B, 1280)

        # Pass through classification head
        logits = self.head(cls_token)  # Shape: (B, num_classes)

        return logits
    
    def head_input_units(self, model_name: str) -> int:
        """
        Get the number of input units for the classification head based on the model name.

        Args:
            model_name (str): Name of the ViT model.

        Returns:
            int: Number of input units for the classification head.
        """
        hidden_units = {
            "phikon": 768,
            "ViT_H": 1280
        }
    
        return hidden_units.get(model_name)
    
    def load_backbone(self, model_name: str) -> nn.Module:
        """
        Load the backbone ViT model based on the model name.

        Args:
            model_name (str): Name of the ViT model to load.

        Returns:
            nn.Module: Loaded ViT model.
        """
        if model_name == 'phikon':
            model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
        elif model_name == 'ViT_H':
            model = ViTModel.from_pretrained('google/vit-huge-patch14-224-in21k')
        else:
            raise ValueError(f"Model {model_name} not implemented")
        return model


class Classifier(nn.Module):
    def __init__(self, model_name='resnet50', lora: bool = False, num_classes: int = 2):
        """
        Classification Model supporting multiple architectures including ResNet, Hoptimus, and Virchow.

        Args:
            model_name (str): Name of the model architecture (e.g., 'resnet50', 'hoptimus', 'virchow').
            lora (bool): Whether to apply LoRA adaptation.
            num_classes (int): Number of output classes.
        """
        super().__init__()
        self.model_name = model_name
        self.lora = lora
        self.num_classes = num_classes
        self.model = self.load_model(model_name, lora, num_classes=num_classes)
        
    def load_model(self, model_name: str, lora: bool = False, num_classes: int = 2) -> nn.Module:
        """
        Load a specified model for classification.
        Args:
            model_name (str): The name of the model to load. Supported models include:
            "ViT_H", "phikon", "resnet50", "hoptimus", "virchow", "virchow2", "uni", "gigapath".
            lora (bool, optional): If True, initialize the model with LoRA (Low-Rank Adaptation). Defaults to False.
            num_classes (int, optional): The number of classes for the classification task. Defaults to 2.
        Returns:
            nn.Module: The initialized model ready for classification.
        Raises:
            ValueError: If the specified model_name is not implemented.
        """
        model_map = {
            "ViT_H": lambda: ViTForClassification(model_name="ViT_H", num_classes=num_classes),
            "phikon": lambda: ViTForClassification(model_name="phikon", num_classes=num_classes),
            "resnet50": lambda: load_model_and_transforms("resnet50")[0],
            "hoptimus": lambda: load_model_and_transforms("hoptimus")[0],
            "virchow": lambda: load_model_and_transforms("virchow")[0],
            "virchow2": lambda: load_model_and_transforms("virchow2")[0],
            "uni": lambda: load_model_and_transforms("uni")[0],
            "gigapath": lambda: load_model_and_transforms("gigapath")[0]
        }
        
        model = model_map.get(model_name, None)
        
        if model is None:
            raise ValueError(f"Model {model_name} not implemented")
        
        model = model()
        # Add classification head if not present for ViT models
        if model_name not in VIT_FOR_CLASSIFICATION:
            model = self.initialize_classification_head(model_name, model, num_classes)

        if lora:
            model = self.initialize_lora_model(model_name, model)

        return model
        
    def initialize_lora_model(self, model_name: str, model: nn.Module) -> nn.Module:
        """
        Initialize a LoRA (Low-Rank Adaptation) model with the given configuration.
        Args:
            model_name (str): The name of the model to be initialized.
            model (nn.Module): The base model to which the LoRA configuration will be applied.
        Returns:
            nn.Module: The model with the LoRA configuration applied.
        Raises:
            ValueError: If no LoRA configuration is available for the given model_name.
        """
        if model_name not in VALID_LORA_MODULES:
            raise ValueError(f"No LoRA configuration available for {model_name}")
        
        config = peft.LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=VALID_LORA_MODULES[model_name],
            lora_dropout=0.1,
            bias="none",
            modules_to_save=["head"]
        )

        model = peft.get_peft_model(model, config)
        print(f"Initialized LoRA model for {model_name}")
        model.print_trainable_parameters()
        return model

    def initialize_classification_head(self, model_name: str, model: nn.Module, num_classes: int = 2) -> nn.Module:
        """
        Initializes the classification head for different models.

        Args:
            model_name (str): The name of the model (e.g., "resnet50", "hoptimus").
            model (nn.Module): The model to modify.
            num_classes (int, optional): Number of output classes. Defaults to 2.

        Returns:
            nn.Module: The model with the updated classification head.
        """
        num_classes = 1 if num_classes == 2 else num_classes
        
        if model_name == 'resnet50':
            model.fc = torch.nn.Linear(2048, num_classes)
            
        elif model_name == 'hoptimus':
            model.head = torch.nn.Linear(1536, num_classes)
        
        elif model_name == 'virchow' or model_name == 'virchow2' or model_name == 'ViT_H':
            model.head = torch.nn.Linear(1280, num_classes)
        
        elif model_name == 'uni':
            model.head = torch.nn.Linear(1024, num_classes)
            
        elif model_name == 'gigapath':
            model.head = torch.nn.Linear(1536, num_classes)
        else:
            raise ValueError(f"Classification head initialization not implemented for model: {model_name}")
        
        return model
    
    def load_pretrained_model(self, checkpoint_path: str) -> nn.Module:
        """
        Load a pretrained model from a checkpoint file.

        Args:
            checkpoint_path (str): The path to the checkpoint file. For LoRA models, this should be the directory where the weights are stored.

        Returns:
            nn.Module: The loaded model.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
            RuntimeError: If there is an error loading the model from the checkpoint.
        """
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        try:
            if self.lora:
                model = self.load_pretrained_lora_model(self.model_name, checkpoint_path)
            else:
                model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {checkpoint_path}: {str(e)}")


    def forward(self, x):
        """
        Forward pass with conditional processing for different architectures.

        Args:
            x (Tensor): Tensor with shape [B, 3, W, H] (image batch).

        Returns:
            Tuple[Tensor]: logits, probabilities, and binary labels.
        """
        output = self.model(x)

        # Handle different model architectures
        if self.model_name in ['virchow', 'virchow2']:
            # Virchow outputs multiple tokens, use only class token (assuming it's at index 0)
            logits = output[:, 0, :]  # Shape: (B, D) using only [CLS] token

        else:
            logits = output  # Default case for ResNet, Hoptimus, etc.

        logits = logits.squeeze()
        Y_prob = torch.sigmoid(logits)
        Y_hat = (Y_prob > 0.5).float()
        
        return logits, Y_prob, Y_hat
