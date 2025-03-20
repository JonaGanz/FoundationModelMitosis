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
from src.classifier import Classifier

BATCHE_SIZE = 16
NUM_WORKERS = 4
PATCH_SIZE = 224
DEVICE = 'cuda'
TEST_PORTION = 0.2
PSEUDO_EPOCH_LENGTH = 1280
LR = 1e-4
N_EPOCHS = 100


def write_args_to_yaml(args, path):
    with open(path, 'w') as f:
        yaml.dump(vars(args), f)

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
    parser.add_argument('--lora', action='store_true', default=False)
    parser.add_argument('--model_name', type=str, default='resnet50')
    parser.add_argument('--debug', action='store_true', default=False)
    return parser.parse_args()

def main(args):
    
    torch.set_float32_matmul_precision('medium')
    
    # get tumor types
    # load the csv file
    df = pd.read_csv(args.path_to_csv_file)
    assert 'tumortype' in df.columns
    tumor_types = df['tumortype'].unique()
    for tumor_type in tumor_types:
        # loop over seeds

        # make a directory to save the results if it does not exist
        out_path = Path(f"{args.checkpoint_path}/{args.exp_code}/{tumor_type}")
        out_path.mkdir(parents=True, exist_ok=True)
        write_args_to_yaml(args, f"{args.checkpoint_path}/{args.exp_code}/args.yaml")
        
        for run_idx, seed in enumerate([42,43,44,45,46]):
            print(f"##### Starting run {run_idx} with seed {seed} and tumor_type {tumor_type} #####")
            

            # initialize tensorboard logger
            # Initialize TensorBoard writer
            log_path = Path(f"{args.checkpoint_path}/{args.exp_code}/{tumor_type}/{run_idx}")
            log_path.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir=log_path)
            
            
            np.random.seed(seed)
            torch.manual_seed(seed)
            # load model
            model = Classifier(args.model_name, args.lora)
            model.cuda()
            # select all samples of one type as training samples
            train_indice = df[df['tumortype'] == tumor_type].index
            # select 20% of the train_inidce as further test samples
            test_indice = df.loc[train_indice].sample(frac=args.test_portion, replace=False).index
            # also sample the samples of all instances of another tumor type as test samples
            test_indice = test_indice.union(df[df['tumortype'] != tumor_type].index)
            
            
            # remove test samples from train samples
            train_indice = df.drop(test_indice).index
            # select validation samples
            test_indice = df.drop(train_indice).index                
            test_df = df.loc[test_indice]

            # select validation samples
            train_df = df.loc[train_indice]
            val_df = train_df.sample(frac=args.test_portion, replace=False)
            train_df = train_df.drop(val_df.index)
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
            
            if args.debug:
                # only hold 10 test samples, drop the rest, use normal df
                test_df = df[df['split'] == 'test'].head(7)
                df.drop(df[df['split'] == 'test'].index, inplace=True)
                df = pd.concat([df, test_df])
                  
            # initialize dataloaders
            base_ds = Mitosis_Base_Dataset(
                csv_file=df,
                image_dir=Path(args.image_dir),
            )
            
            base_transform = model.input_transform
            
            train_transform = T.Compose([
                T.RandomApply([T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)], p=0.5),
                T.RandomApply([T.GaussianBlur(kernel_size=(5,5), sigma=(0.1, 1))], p=0.1),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomApply([T.RandomRotation(degrees=360)], p=0.5),
                *model.input_transform.transforms,
            ])
                
            
            train_ds = base_ds.return_split(
                split = 'train',
                patch_size = args.patch_size,
                level = 0,
                transforms = train_transform,
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
                batch_size=BATCHE_SIZE,
                shuffle=True,
                num_workers=NUM_WORKERS,
                collate_fn=train_ds.collate_fn
            )
            val_loader = torch.utils.data.DataLoader(
                val_ds,
                batch_size=BATCHE_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                collate_fn=val_ds.collate_fn
            )
            test_loader = torch.utils.data.DataLoader(
                test_ds,
                batch_size=BATCHE_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
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
                        if args.lora:
                            model.model.save_pretrained(f"{out_path}/{run_idx}")
                        else:
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
                        if args.lora:
                            model.model.save_pretrained(f"{out_path}/{run_idx}")
                        else:
                            best_model = model.state_dict()
                    
                # resample training patches
                train_loader.dataset.resample_patches()
            # save best model
            if args.lora:
                # for lora models, the best model has been saved to disc during training.
                pass
            else:    
                torch.save(best_model, f"{out_path}/{run_idx}.pth")
            
            if best_model is not None:
                if args.lora:
                    model.load_pretrained_lora_model(args.model_name, f"{out_path}/{run_idx}")
                    model.cuda()
                else:
                    model.load_state_dict(best_model)
        
            # Close the TensorBoard writer
            writer.close()
            
            # test the model
            results = test(model, test_loader)
            results.to_csv(f"{out_path}/{run_idx}_results.csv")
            
            print(f"##### Run {run_idx} finished #####")
        
    print("All runs finished")
        

if __name__ == "__main__":
    args = get_args()
    main(args)