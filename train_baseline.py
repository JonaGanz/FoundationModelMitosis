import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path
from tqdm import tqdm
from MeningiomaDataset.src.classification_dataset import Mitosis_Base_Dataset
from src.utils import extract_patch_features_from_dataloader, load_model_and_transforms, collate_fn, return_forward
import torch
import pickle
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.tensorboard import SummaryWriter
from src.utils import collate_fn
from torchvision import transforms as T
from torch.optim.lr_scheduler import OneCycleLR
import yaml
from torch.nn.utils import clip_grad_norm_
import timm

BATCHE_SIZE = 16
NUM_WORKERS = 4
PATCH_SIZE = 224
DEVICE = 'cuda'
TEST_PORTION = 0.2
PSEUDO_EPOCH_LENGTH = 1280
LR = 1e-4
N_EPOCHS = 100

def load_model(model_name):
    if model_name == 'resnet50':
        model = resnet50(weights = ResNet50_Weights.DEFAULT)
        model.fc = torch.nn.Linear(2048, 1)
        
    elif model_name == 'hoptimus':
        model = timm.create_model(
        "hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5, dynamic_img_size=False)
        model.head = torch.nn.Linear(1536, 1)
        
        # freeze erverything except the head
        for name, param in model.named_parameters():
            if 'head' not in name:
                param.requires_grad = False
        
    else:
        raise ValueError(f"Model {model_name} not implemented")
    
    return model

def write_args_to_yaml(args, path):
    with open(path, 'w') as f:
        yaml.dump(vars(args), f)

class MitosisClassifier(torch.nn.Module):
    def __init__(self, model_name = 'resnet50'):
        super().__init__()
        self.model = load_model(model_name)
        
    def forward(self, x):
        """Foward pass

        Args:
            x (Tensor): Tensor with shape [B, 3, W, H]

        Returns:
            Tuple[Tensor]: logits, probabilities and labels
        """
        logits = self.model(x)
        logits = logits.squeeze()
        Y_prob = torch.sigmoid(logits)
        Y_hat = (Y_prob > 0.5).float()
        return logits, Y_prob, Y_hat

def train_one_epoch(model, optimizer, criterion, train_loader, scheduler = None, clip_grad = False):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    for images, labels in tqdm(train_loader):
        images = images.cuda()
        labels = labels.cuda()
        
        optimizer.zero_grad()
        logits, _, Y_hat = model(images) 
        loss = criterion(logits, labels.float())
        if clip_grad:
            clip_grad_norm_(model.parameters(), 0.1)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        running_loss += loss.item()
        total += labels.size(0)
        correct += (Y_hat == labels).sum().item()
        
    return running_loss / len(train_loader), correct / total

def validate(model, criterion, val_loader):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images = images.cuda()
            labels = labels.cuda()

        logits, _, Y_hat = model(images) 
        loss = criterion(logits, labels.float())        
        running_loss += loss.item()
        total += labels.size(0)
        correct += (Y_hat == labels).sum().item()
        
    return running_loss / len(val_loader), correct / total

def test(model, test_loader):
    # forwards the images through the model and saves the predictions
    results = []
    with torch.no_grad():
        for images, labels, files, coords in tqdm(test_loader):
            images = images.cuda()
            labels = labels.cuda()
            logits, Y_prob, Y_hat = model(images) 
            
            for file, coord, label, pred, output in zip(files, coords, labels.cpu().numpy(), Y_hat.cpu().numpy(), Y_prob.cpu().numpy()):
                results.append({
                    'files': file,
                    'x': coord[0],
                    'y': coord[1],
                    'labels': label,
                    'predicted': pred,
                    'pobs': output
                })
    results = pd.DataFrame(results)
    results.columns = ['file', 'x', 'y', 'label', 'predicted', 'probs']
    
    return results
        
def parse_float_list(s):
    return [float(item) for item in s.split(',')]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_csv_file', type=str, required=True)
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--exp_code', type=str, default='None')
    parser.add_argument('--equalize', type=bool, default=False)
    parser.add_argument('--scheduler', action='store_true', default=False)
    parser.add_argument('--early_stopping', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=BATCHE_SIZE)
    parser.add_argument('--num_workers', type=int, default=NUM_WORKERS)
    parser.add_argument('--patch_size', type=int, default=PATCH_SIZE)
    parser.add_argument('--test_portion', type=float, default=TEST_PORTION)
    parser.add_argument('--pseudo_epoch_length', type=int, default=PSEUDO_EPOCH_LENGTH)
    parser.add_argument('--learning_rate', type=float, default=LR)
    parser.add_argument('--num_epochs', type=int, default=N_EPOCHS)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--gradient_clipping', type=bool, default=False)
    parser.add_argument('--train_sizes', type=parse_float_list, default=[0.001, 0.01, 0.1, 1.0])
    parser.add_argument('--model_name', type=str, default='resnet50')
    parser.add_argument('--augmentation', type=bool, default=True)
    return parser.parse_args()

def main(args):
    
    # loop over seeds
    for train_size in args.train_sizes:
        # make a directory to save the results if it does not exist
        out_path = Path(f"{args.checkpoint_path}/{args.exp_code}/{train_size}")
        out_path.mkdir(parents=True, exist_ok=True)
        # write args to yaml
        write_args_to_yaml(args, f"{args.checkpoint_path}/{args.exp_code}/args.yaml")
        for run_idx, seed in enumerate([42, 43, 44, 45, 46]):
            print(f"##### Starting run {run_idx} with seed {seed} and train_size {train_size} #####")
            
            # load the csv file
            df = pd.read_csv(args.path_to_csv_file)
            # initialize tensorboard logger
            # Initialize TensorBoard writer
            log_path = Path(f"{args.checkpoint_path}/{args.exp_code}/{train_size}/{run_idx}")
            log_path.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir=log_path)
            
            
            np.random.seed(seed)
            torch.manual_seed(seed)
            # load model
            model = MitosisClassifier(args.model_name)
            model.cuda()
            # split data into train and test
            test_indice = np.random.choice(df.index, int(len(df)*args.test_portion), replace=False)                
            test_df = df.loc[test_indice]
            train_df = df.drop(test_indice)
            # slelect training samples based on train_size
            train_indice = np.random.choice(train_df.index, int(len(train_df)*train_size), replace=False)
            train_df = train_df.loc[train_indice]
            
            # all_train_indice = train_df.index
            # if args.equalize:
            #     all_train_labels = train_df['label']
            #     # get the positive and negative samples
            #     positive_samples = all_train_indice[all_train_labels == 1]
            #     negative_samples = all_train_indice[all_train_labels == 0]
                
            #     min_samples = min(len(positive_samples), len(negative_samples))
            #     positive_indice = np.random.choice(positive_samples, min_samples, replace=False)
            #     negative_samples = np.random.choice(negative_samples, min_samples, replace=False)
            #     all_train_indice = np.concatenate([positive_indice, negative_samples])
            #     train_df = train_df.loc[all_train_indice]
            
            # select validation data
            val_indice = np.random.choice(train_df.index, int(len(train_df)*args.test_portion), replace=False)
            val_df = train_df.loc[val_indice]
            train_inidice = train_df.drop(val_indice).index
            train_df = train_df.loc[train_inidice]
            # check dfs for overlaps
            assert len(set(train_df.index).intersection(set(val_df.index))) == 0
            assert len(set(train_df.index).intersection(set(test_df.index))) == 0
            assert len(set(val_df.index).intersection(set(test_df.index))) == 0
            # assing a column split to the df
            df['split'] = 'NONE'
            df.loc[train_df.index, 'split'] = 'train'
            df.loc[val_df.index, 'split'] = 'val'
            df.loc[test_df.index, 'split'] = 'test'
            
            df.to_csv(f"{out_path}/{run_idx}_split.csv")  
            # initialize dataloaders
            base_ds = Mitosis_Base_Dataset(
                csv_file=df,
                image_dir=Path(args.image_dir),
            )
            
            base_transform = T.Compose([
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            train_transform = T.Compose([
                T.RandomApply([T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)], p=0.5),
                T.RandomApply([T.GaussianBlur(kernel_size=(5,5), sigma=(0.1, 1))], p=0.1),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomApply([T.RandomRotation(degrees=360)], p=0.5),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
                
            
            train_ds = base_ds.return_split(
                split = 'train',
                patch_size = args.patch_size,
                level = 0,
                transforms = train_transform if args.augmentation else base_transform,
                pseudo_epoch_length = args.pseudo_epoch_length
            )
            val_ds = base_ds.return_split(
                split = 'val',
                patch_size = args.patch_size,
                level = 0,
                transforms = base_transform,
                pseudo_epoch_length = args.pseudo_epoch_length
            )
            test_ds = base_ds.return_split(
                split = 'test',
                patch_size = args.patch_size,
                level = 0,
                transforms = base_transform
            )
            
            train_loader = torch.utils.data.DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                collate_fn=train_ds.collate_fn
            )
            val_loader = torch.utils.data.DataLoader(
                val_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=val_ds.collate_fn
            )
            test_loader = torch.utils.data.DataLoader(
                test_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=collate_fn
            )
            # initialize optimizer and criterion
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            criterion = torch.nn.BCEWithLogitsLoss()
            
            if args.scheduler:
                scheduler = OneCycleLR(optimizer, max_lr=args.learning_rate, steps_per_epoch=len(train_loader), epochs=args.num_epochs)
            else:
                scheduler = None
            # train the model with early stopping
            best_loss = np.inf
            patience = args.patience
            trigger_times = 0
            best_model = None

            for epoch in range(args.num_epochs):
                train_loss, train_acc = train_one_epoch(model, optimizer, criterion, train_loader, scheduler, args.gradient_clipping)
                val_loss, val_acc = validate(model, criterion, val_loader)
                print(f"Epoch: {epoch}, Train loss: {train_loss}, Train acc: {train_acc}, Val loss: {val_loss}, Val acc: {val_acc}")

                # Log metrics to TensorBoard
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/val', val_loss, epoch)
                writer.add_scalar('Accuracy/train', train_acc, epoch)
                writer.add_scalar('Accuracy/val', val_acc, epoch)
                writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
                if args.early_stopping:
                    if val_loss < best_loss:
                        print(f"Loss improved from {best_loss} to {val_loss}, saving model and resetting trigger times")
                        best_loss = val_loss
                        best_model = model.state_dict()
                        trigger_times = 0
                    else:
                        trigger_times += 1
                        print(f"Early Stopping Counter: {trigger_times}/{patience}")

                    if trigger_times >= patience:
                        print("Early stopping!")
                        break
                else:
                    if val_loss < best_loss:
                        print(f"Loss improved from {best_loss} to {val_loss}, saving model")
                        best_loss = val_loss
                        best_model = model.state_dict()
                    
                # resample training patches
                train_loader.dataset.resample_patches()
            # save best model
            torch.save(best_model, f"{out_path}/{run_idx}.pth")
            
            if best_model is not None:
                model.load_state_dict(best_model)
        
            # Close the TensorBoard writer
            writer.close()
            
            # test the model
            results = test(model, test_loader)
            results.to_csv(f"{out_path}/{run_idx}_results.csv")
            print(np.unique(train_df['label'], return_counts=True))
            print(f"##### Run {run_idx} finished #####")
        
    print("All runs finished")
        

if __name__ == "__main__":
    args = get_args()
    # save args
    main(args)