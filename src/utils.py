import numpy as np
import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision import transforms as T
import timm
from tqdm import tqdm
from transformers import ViTModel

from torchvision.models import resnet50, ResNet50_Weights




# adapted from UNI library
@torch.no_grad()
def extract_patch_features_from_dataloader(model, dataloader, forward_fn):
    """Uses model to extract features+labels from images iterated over the dataloader.

    Args:
        model (torch.nn): torch.nn CNN/VIT architecture with pretrained weights that extracts d-dim features.
        dataloader (torch.utils.data.DataLoader): torch.utils.data.DataLoader object of N images.

    Returns:
        dict: Dictionary object that contains (1) [N x D]-dim np.array of feature embeddings, and (2) [N x 1]-dim np.array of labels

    """
    all_embeddings, all_labels, all_files, all_coords = [], [], [], []
    batch_size = dataloader.batch_size
    device = next(model.parameters())[0].device

    for batch_idx, (batch, target, files, coords) in tqdm(
        enumerate(dataloader), total=len(dataloader)
    ):
        remaining = batch.shape[0]
        if remaining != batch_size:
            _ = torch.zeros((batch_size - remaining,) + batch.shape[1:]).type(
                batch.type()
            )
            batch = torch.vstack([batch, _])

        batch = batch.to(device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            embeddings = forward_fn(model, batch, remaining)
            # to work with detection datasets
            if isinstance(target, list):
                labels = np.array([1 if t['labels'].sum() > 0 else 0 for t in target])[:remaining]
            else:
                labels = target.numpy()[:remaining]
            assert not torch.isnan(embeddings).any()

        all_embeddings.append(embeddings)
        all_labels.append(labels)
        all_files.extend(files)
        all_coords.append(coords)

    asset_dict = {
        "embeddings": np.vstack(all_embeddings).astype(np.float32),
        "labels": np.concatenate(all_labels),
        "files": all_files,
        "coords": np.vstack(all_coords)
    }

    return asset_dict

def return_forward(model_name: str) -> callable:
    """
    Returns the appropriate forward function for the given model name.

    Args:
        model_name (str): The name of the model.

    Returns:
        callable: A function that takes a model, a batch of images, and the remaining number of images,
                  and returns the extracted embeddings.
    """
    if model_name == 'uni':
        def forward_fn(model, batch, remaining):
            embeddings = model(batch).detach().cpu()[:remaining, :].cpu()
            return embeddings
        
    elif model_name in ['virchow', 'virchow2']:
        def forward_fn(model, batch, remaining):
            output = model(batch).detach().cpu()[:remaining, :].cpu()
            class_token = output[:, 0]
            patch_token = output[:, 1:]
            embeddings = torch.cat([class_token, patch_token.mean(dim=1)], dim=-1)
            return embeddings
        
    elif model_name == 'phikon':
        def forward_fn(model, batch, remaining):
            embeddings = model(batch).last_hidden_state[:, 0, :].detach().cpu()[:remaining, :].cpu()
            return embeddings
        
    elif model_name in ['resnet50', 'gigapath', 'hoptimus']:
        def forward_fn(model, batch, remaining):
            embeddings = model(batch).detach().cpu()[:remaining, :].cpu()
            return embeddings

    elif model_name == 'ViT_H':
        def forward_fn(model, batch, remaining):
            output = model(batch).last_hidden_state.detach().cpu()[:remaining, :].cpu()
            class_token = output[:, 0]
            patch_token = output[:, 1:]
            embeddings = torch.cat([class_token, patch_token.mean(dim=1)], dim=-1)
            return embeddings
    
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")
    
    return forward_fn

# initialize UNI
def load_model_and_transforms(model_name: str) -> tuple:
    """
    Loads the specified model and its corresponding transforms.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        tuple: A tuple containing the model and the transforms.
    """
    if model_name == 'uni':
        model = timm.create_model(
            "vit_large_patch16_224",
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True
        )
        model.load_state_dict(torch.load("UNI/checkpoints/UNI/pytorch_model.bin"))
        transforms = T.Compose([
            T.Resize(224),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    
    elif model_name == 'virchow':
        model = timm.create_model(
            "hf-hub:paige-ai/Virchow",
            pretrained=False,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU)
        model.load_state_dict(torch.load("checkpoints/Virchow/pytorch_model.bin"))
        transforms = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    
    elif model_name == 'virchow2':
        model = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=False,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU)
        model.load_state_dict(torch.load("checkpoints/Virchow2/pytorch_model.bin"))
        transforms = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        
    elif model_name == 'phikon':
        model = ViTModel.from_pretrained(
            "owkin/phikon",
            add_pooling_layer=False,
        )
        transforms = T.Compose([
            T.Resize(224),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    
    elif model_name == 'ctranspath':
        raise NotImplementedError
        # model = ctran.ctranspath()
        # model.head = torch.nn.Identity()
        # td = torch.load('checkpoints/CTransPath/ctranspath.pth')
        # model.load_state_dict(td['model'], strict = True)
        # transforms = T.Compose([
        #     T.Resize(256),
        #     T.ToTensor(),
        #     T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        # ])
    
    elif model_name == 'resnet50':
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.fc = torch.nn.Identity()
        transforms = T.Compose([
            T.Resize(224),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    
    elif model_name == 'ViT_H':
        model = ViTModel.from_pretrained('google/vit-huge-patch14-224-in21k')
        # normalization according to https://huggingface.co/google/vit-huge-patch14-224-in21k
        transforms = T.Compose([
            T.Resize(224),
            T.ToTensor(),
            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
        
    elif model_name == 'gigapath':
        model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
        transforms = T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        
    elif model_name == 'hoptimus':
        model = timm.create_model(
            "hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5, dynamic_img_size=False
        )
        transforms = T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.707223, 0.578729, 0.703617), std=(0.211883, 0.230117, 0.177517)),
        ])
        
    else:
        raise ValueError(f"Model {model_name} not implemented")
    
    return model, transforms


def collate_fn(batch):
    """Collate function for the data loader."""
    images = list()
    targets = list()
    files = list()
    coords = list()
    for b in batch:
        images.append(b[0])
        targets.append(b[1])
        files.append(b[2]) 
        coords.append(b[3])
    images = torch.stack(images, dim=0)
    targets = torch.tensor(targets, dtype=torch.int64)
    return images, targets, files, coords