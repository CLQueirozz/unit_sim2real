"""
Training script for 2D Semantic Segmentation using only RGB and Semantic masks.
This script uses a pre-trained DeepLabV3 model for a more direct 2D approach.
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
import cv2
import glob
import segmentation_models_pytorch as smp
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
CONFIG = {
    'data_dir': 'data/dataset/soybean',
    'num_classes': 5,  # ground, leaf, branch, weed, obstacle
    'class_names': {0: "ground", 1: "leaf", 2: "branch", 3: "weed", 4: "obstacle"},
    'batch_size': 4,
    'epochs': 50,
    'learning_rate': 1e-4,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'ignore_index': -1, # Value in the semantic mask to be ignored during loss calculation
    'data_augmentation': True, # Enable/disable data augmentation
    'model_save_path': 'unet_efficientnetb3_semantic.pth',
}

# --- DATASET ---
class ImageSemanticDataset(Dataset):
    """
    A simplified dataset that loads only RGB images and their corresponding
    2D semantic segmentation masks.
    """
    def __init__(self, root_dir, num_classes=5, transform=None, data_augmentation=False):
        self.root_dir = root_dir
        self.transform = transform
        self.num_classes = num_classes
        self.data_augmentation = data_augmentation
        self.rgb_files = sorted(glob.glob(os.path.join(root_dir, 'rgb', '*.png')))
        if not self.rgb_files:
            self.rgb_files = sorted(glob.glob(os.path.join(root_dir, 'rgb', '*.jpg')))

        if not self.rgb_files:
            raise ValueError(f"No images found in {os.path.join(root_dir, 'rgb')}")

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb_path = self.rgb_files[idx]
        base_name = os.path.splitext(os.path.basename(rgb_path))[0]

        # Construct path to semantic mask
        # The folder name in the original dataset is 'camera_semantic'
        semantic_path = os.path.join(self.root_dir, 'camera_semantic', f"{base_name}.png")

        # Load RGB Image
        rgb_img = cv2.imread(rgb_path)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        rgb_tensor = TF.to_tensor(rgb_img)

        # Load Semantic Mask
        semantic_mask = cv2.imread(semantic_path, cv2.IMREAD_UNCHANGED)
        if semantic_mask is None:
            raise FileNotFoundError(f"Semantic mask not found: {semantic_path}")

        # Ensure mask is single channel
        if semantic_mask.ndim == 3:
            semantic_mask = semantic_mask[:, :, 0]

        semantic_tensor = torch.from_numpy(semantic_mask.astype(np.int64))

        # Apply transformations if any
        if self.transform:
            rgb_tensor, semantic_tensor = self.transform(rgb_tensor, semantic_tensor)

        # Apply data augmentation if enabled
        if self.data_augmentation:
            # Random horizontal flip
            if torch.rand(1) > 0.5:
                rgb_tensor = TF.hflip(rgb_tensor)
                semantic_tensor = TF.hflip(semantic_tensor)

            # Color Jitter (only on RGB)
            if torch.rand(1) > 0.5:
                color_jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
                rgb_tensor = color_jitter(rgb_tensor)

            # Noise
            if torch.rand(1) > 0.5:
                noise = torch.randn_like(rgb_tensor) * 0.1
                rgb_tensor = torch.clamp(rgb_tensor + noise, 0, 1)

            # Blur
            if torch.rand(1) > 0.5:
                rgb_tensor = TF.gaussian_blur(rgb_tensor, kernel_size=5, sigma=(0.1, 2.0))



        return {'rgb': rgb_tensor, 'semantic': semantic_tensor, 'frame_id': base_name}

# --- METRICS ---
def compute_metrics(pred, target, num_classes):
    """Computes Pixel Accuracy and Mean IoU, ignoring -1 labels."""
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    
    # Create a mask to ignore pixels with the ignore_index value
    valid_mask = (target != CONFIG['ignore_index'])
    
    # Pixel Accuracy
    correct = np.sum((pred == target) & valid_mask)
    total = np.sum(valid_mask)
    pixel_acc = correct / total

    # Mean IoU
    iou_list = []
    for c in range(num_classes):
        tp = np.sum((pred == c) & (target == c))
        fp = np.sum((pred == c) & (target != c))
        fn = np.sum((pred != c) & (target == c) & valid_mask) # Only count valid target pixels as false negatives
        
        union = tp + fp + fn 
        if union == 0:
            iou = 0.0 # Or float('nan')
        else:
            iou = tp / union
        iou_list.append(iou)
        
    miou = np.nanmean(iou_list)
    return pixel_acc, miou

# --- TRAINING AND VALIDATION ---
def train_one_epoch(model, dataloader, criterion, optimizer, device, num_classes):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_miou = 0.0

    for batch in tqdm(dataloader, desc="Training"):
        rgb = batch['rgb'].to(device)
        semantic = batch['semantic'].to(device)

        optimizer.zero_grad()
        
        # DeepLabV3 returns a dictionary
        output = model(rgb)
        loss = criterion(output, semantic)
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        preds = torch.argmax(output, dim=1)
        acc, miou = compute_metrics(preds, semantic, num_classes)
        total_acc += acc
        total_miou += miou

    return total_loss / len(dataloader), total_acc / len(dataloader), total_miou / len(dataloader)

def validate_one_epoch(model, dataloader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_miou = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            rgb = batch['rgb'].to(device)
            semantic = batch['semantic'].to(device)

            output = model(rgb)
            loss = criterion(output, semantic)

            total_loss += loss.item()
            
            preds = torch.argmax(output, dim=1)
            acc, miou = compute_metrics(preds, semantic, num_classes)
            total_acc += acc
            total_miou += miou

    return total_loss / len(dataloader), total_acc / len(dataloader), total_miou / len(dataloader)


def main():
    print(f"Using device: {CONFIG['device']}")

    # --- DATA ---
    # Create separate datasets for training and validation
    # Augmentation should only be applied to the training set.
    train_dataset = ImageSemanticDataset(
        CONFIG['data_dir'], 
        num_classes=CONFIG['num_classes'],
        data_augmentation=CONFIG['data_augmentation']
    )
    val_dataset = ImageSemanticDataset(
        CONFIG['data_dir'], 
        num_classes=CONFIG['num_classes'],
        data_augmentation=False # No augmentation for validation
    )
    
    # Create a random split of indices, then create subsets
    indices = torch.randperm(len(train_dataset)).tolist()
    train_indices = indices[:int(0.8 * len(train_dataset))]
    val_indices = indices[int(0.8 * len(train_dataset)):]
    
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2)

    print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples.")

    # --- MODEL ---
    # Load a U-Net model with a pre-trained EfficientNet encoder
    print("Loading U-Net with EfficientNet encoder...")
    model = smp.Unet(
        encoder_name="efficientnet-b3",
        encoder_weights="imagenet",
        in_channels=3,
        classes=CONFIG['num_classes'],
    )
    model = model.to(CONFIG['device'])

    # --- LOSS & OPTIMIZER ---
    # CrossEntropyLoss is standard for multi-class segmentation
    criterion = nn.CrossEntropyLoss(ignore_index=CONFIG['ignore_index'])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    # --- TRAINING LOOP ---
    best_val_miou = 0.0

    for epoch in range(CONFIG['epochs']):
        print(f"\n--- Epoch {epoch+1}/{CONFIG['epochs']} ---")

        train_loss, train_acc, train_miou = train_one_epoch(
            model, train_loader, criterion, optimizer, CONFIG['device'], CONFIG['num_classes']
        )
        print(f"Train -> Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, mIoU: {train_miou:.4f}")

        val_loss, val_acc, val_miou = validate_one_epoch(
            model, val_loader, criterion, CONFIG['device'], CONFIG['num_classes']
        )
        print(f"Validation -> Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, mIoU: {val_miou:.4f}")

        # Save the best model based on validation mIoU
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            torch.save(model.state_dict(), CONFIG['model_save_path'])
            print(f"✨ New best model saved to {CONFIG['model_save_path']} (mIoU: {best_val_miou:.4f})")

    print("\n--- Training Complete ---")
    print(f"Best validation mIoU: {best_val_miou:.4f}")
    print(f"Model saved to {CONFIG['model_save_path']}")

    # --- VISUALIZE A PREDICTION ---
    model.load_state_dict(torch.load(CONFIG['model_save_path']))
    model.eval()
    sample = val_dataset[np.random.randint(len(val_dataset))]
    rgb_tensor = sample['rgb'].unsqueeze(0).to(CONFIG['device'])
    
    with torch.no_grad():
        pred_mask = model(rgb_tensor)
        pred_mask = torch.argmax(pred_mask, dim=1).squeeze(0).cpu().numpy()

    rgb_img = sample['rgb'].permute(1, 2, 0).numpy()
    true_mask = sample['semantic'].numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(rgb_img)
    axes[0].set_title("Input RGB")
    axes[0].axis('off')

    axes[1].imshow(true_mask, cmap='tab10', vmin=0, vmax=CONFIG['num_classes']-1)
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis('off')

    axes[2].imshow(pred_mask, cmap='tab10', vmin=0, vmax=CONFIG['num_classes']-1)
    axes[2].set_title("Predicted Mask")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('2d_semantic_prediction_example.png')
    print("\nSaved a sample prediction to '2d_semantic_prediction_example.png'")


if __name__ == '__main__':
    main()