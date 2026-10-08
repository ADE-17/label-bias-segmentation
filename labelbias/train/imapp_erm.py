import os
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import segmentation_models_pytorch as smp

from labelbias.data.imapp import IMAPPBiasedDataset
from labelbias.metrics import combined_loss
from sklearn.model_selection import KFold

'''
python -m labelbias.train.imapp_erm \
  --csv_path configs/imapp_processed.csv \
  --epochs 10 \
  --batch_size 32 \
  --bias_ratio 1.0 \
  --biased_skin_tones 2 3 \
  --output_dir /path/to/output \
  --exp_name imapp_bias_baseline \
  --fold 0
'''

class IMAPPMetrics:
    """Custom metrics tracker tailored for 4 skin tone categories."""
    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.classes = [0, 1, 2, 3]  # Very Light, Light, Intermediate, Tan
        self.names = ['Very Light', 'Light', 'Intermediate', 'Tan']
        self.reset()
        
    def reset(self):
        self.total_inter = np.zeros(self.num_classes)
        self.total_union = np.zeros(self.num_classes)
        
        self.group_inter = {c: np.zeros(self.num_classes) for c in self.classes}
        self.group_union = {c: np.zeros(self.num_classes) for c in self.classes}
        
    def update(self, preds: torch.Tensor, targets: torch.Tensor, groups: torch.Tensor):
        preds = preds.cpu().numpy()
        targets = targets.cpu().numpy()
        groups = groups.cpu().numpy()
        
        for i in range(len(preds)):
            p = preds[i]
            t = targets[i]
            g = int(groups[i])
            
            for c in range(self.num_classes):
                pc = (p == c)
                tc = (t == c)
                inter = (pc & tc).sum()
                union = (pc | tc).sum()
                
                self.total_inter[c] += inter
                self.total_union[c] += union
                if g in self.group_inter:
                    self.group_inter[g][c] += inter
                    self.group_union[g][c] += union
                
    def compute(self):
        res = {}
        iou = self.total_inter / np.maximum(self.total_union, 1)
        res['mean_iou'] = np.mean(iou)
        res['iou_foreground'] = iou[1] if self.num_classes > 1 else 0.0
        
        for g in self.classes:
            g_iou = self.group_inter[g] / np.maximum(self.group_union[g], 1)
            name = self.names[g].replace(" ", "_")
            res[f'{name}_iou_foreground'] = g_iou[1] if self.num_classes > 1 else 0.0
            
        # Grouped Eval metrics for Clean (0,1) and Biased (2,3)
        clean_inter = self.group_inter[0] + self.group_inter[1]
        clean_union = self.group_union[0] + self.group_union[1]
        clean_iou = clean_inter / np.maximum(clean_union, 1)
        res['Clean_iou_foreground'] = clean_iou[1] if self.num_classes > 1 else 0.0
        
        biased_inter = self.group_inter[2] + self.group_inter[3]
        biased_union = self.group_union[2] + self.group_union[3]
        biased_iou = biased_inter / np.maximum(biased_union, 1)
        res['Biased_iou_foreground'] = biased_iou[1] if self.num_classes > 1 else 0.0
            
        return res

def parse_args():
    parser = argparse.ArgumentParser(description='Train with human label bias injection on IMA++ 4-class skin tone data.')
    parser.add_argument('--csv_path', type=str, default='configs/imapp_processed.csv', help="Path to imapp_processed.csv")
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--bias_ratio', type=float, default=0.0, help='Set to 0.0 to train without any injected bias')
    parser.add_argument('--biased_skin_tones', type=int, nargs='+', default=[2, 3], help='List of skin tones to bias (e.g. 2 3)')
    parser.add_argument('--output_dir', type=str, default='/path/to/output')
    parser.add_argument('--exp_name', type=str, default='imapp_biased_train')
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Create master dataset to get length
    master_ds = IMAPPBiasedDataset(csv_path=args.csv_path, img_size=(args.img_size, args.img_size), bias_ratio=0.0)
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    splits = list(kf.split(range(len(master_ds))))
    train_idx, val_idx = splits[args.fold]
    
    # Dataloaders
    train_ds = IMAPPBiasedDataset(
        csv_path=args.csv_path, img_size=(args.img_size, args.img_size),
        bias_ratio=args.bias_ratio, biased_skin_tones=args.biased_skin_tones
    )
    # Val uses bias_ratio 0 (clean STAPLE masks everywhere)
    val_ds = IMAPPBiasedDataset(csv_path=args.csv_path, img_size=(args.img_size, args.img_size), bias_ratio=0.0)
    
    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(Subset(val_ds, val_idx), batch_size=args.batch_size, shuffle=False)
    
    # Model Setup
    model = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    exp_dir = Path(args.output_dir) / 'experiments' / args.exp_name / f'fold_{args.fold}'
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config = vars(args)
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    best_iou = 0.0
    for epoch in range(args.epochs):
        model.train()
        train_metrics_obs = IMAPPMetrics()
        train_metrics_true = IMAPPMetrics()
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}")
        
        for batch in pbar:
            images = batch['image'].to(device)
            masks_obs = batch['mask'].to(device)
            masks_true = batch['clean_mask'].to(device)
            genders = batch['gender'].to(device)
            
            outputs = model(images)
            loss, _ = combined_loss(outputs, masks_obs, loss_mode="ce_dice")
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            preds = outputs.argmax(dim=1)
            train_metrics_obs.update(preds, masks_obs, genders)
            train_metrics_true.update(preds, masks_true, genders)
            pbar.set_postfix({'loss': loss.item()})
            
        res_obs = train_metrics_obs.compute()
        res_true = train_metrics_true.compute()
        print(f"Train - Obs IoU: {res_obs['mean_iou']:.4f} | Clean Group IoU: {res_obs['Clean_iou_foreground']:.4f} | Biased Group IoU: {res_obs['Biased_iou_foreground']:.4f}")
        print(f"Train - True IoU: {res_true['mean_iou']:.4f} | Clean Group IoU: {res_true['Clean_iou_foreground']:.4f} | Biased Group IoU: {res_true['Biased_iou_foreground']:.4f}")
        
        # Validation
        model.eval()
        val_metrics = IMAPPMetrics()
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                # Validation always evaluates against clean_mask!
                masks_true = batch['clean_mask'].to(device)
                genders = batch['gender'].to(device)
                
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                val_metrics.update(preds, masks_true, genders)
                
        vres = val_metrics.compute()
        print(f"Val   - True IoU: {vres['mean_iou']:.4f} | Clean Group IoU: {vres['Clean_iou_foreground']:.4f} | Biased Group IoU: {vres['Biased_iou_foreground']:.4f}")
        
        if vres['mean_iou'] > best_iou:
            best_iou = vres['mean_iou']
            # Save format matching load_fold_model
            torch.save({'model_state_dict': model.state_dict()}, exp_dir / 'best_model.pt')
            print(f"  New best! True IoU: {best_iou:.4f}")
            with open(exp_dir / 'test_metrics.json', 'w') as f:
                json.dump(vres, f, indent=2)

if __name__ == '__main__':
    main()
