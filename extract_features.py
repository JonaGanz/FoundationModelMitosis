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

def main(args):
    ### load model ###
    print(f"------ Loading model: {args.model} ------")
    # load model and matching transforms
    model, transforms = load_model_and_transforms(args.model)
    model.to(args.device)
    model.eval()
    print(f"------ Model loaded: {args.model} ------")
    # load csv to pandas dataframe
    print(f"------ Initializing dataloader ------")
    df = pd.read_csv(args.path_to_csv_file)
    # for debugging purposes
    # df = df.iloc[:100]
    
    # assing split column as test, so just all MFs and Imposters got loaded
    df['split'] = 'test'
    # initilize a dataloader
    base_ds = Mitosis_Base_Dataset(
        csv_file=df,
        image_dir=Path(args.image_dir),
    )
    ds = base_ds.return_split(
        split = 'test',
        patch_size = args.patch_size,
        level = 0,
        transforms = transforms
    )
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    print(f"------ Dataloader initialized ------")
    # extract features
    print(f"------ Extracting features ------")
    # the embeddings are different in length and structuring for the different models, hence they need to be extracted differently
    forward_fn = return_forward(args.model)
    out = extract_patch_features_from_dataloader(model, loader, forward_fn)
    out_path = Path(args.out_path) / f"{args.model}_features.pkl"
    print(f"------ Features extracted, saving_to {str(out_path)} ------")
    # save features
    with open(out_path, 'wb') as f:
        pickle.dump(out, f)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_csv_file', type=str, required=True)
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--patch_size', type=int, default=224)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--model', type=str, choices=['uni','conch','virchow','virchow2','phikon','ctranspath','resnet50','ViT_H', 'gigapath', 'hoptimus'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    main(args)