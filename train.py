import os
import argparse
import torch
import time

# Import Mypackage
from BoneMorphoNet import BoneMorphoNetModel, BoneMorphoNetTrainer

def main():
    """
    Main function for training the Bone Morphology Network model
    """
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Bone Morphology Network Training')
    parser.add_argument('--num_classes', type=int, default=20, help='Number of classes')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--wd', type=float, default=5e-2, help='Weight decay')
    parser.add_argument('--data-path', type=str, default="F:/WorkSpaces/ConvNext_RE/ConvNext2/data/val", help='Dataset path')
    parser.add_argument('--weights', type=str, default='./Mypackage/resources/weights/Biomed_CLIP_all_alpha_1=0.7_4w.pth', help='Pretrained weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False, help='Whether to freeze layers')
    parser.add_argument('--device', type=str, default='cuda:0', help='Training device')
    parser.add_argument('--resume', type=bool, default=False, help='Whether to resume training')
    parser.add_argument('--checkpoint-path', type=str, default='', help='Checkpoint path')
    
    # Add additional arguments required by BoneMorphoNetTrainer
    parser.add_argument('--checkpoint-dir', type=str, default='./Mypackage/resources/checkpoint/', help='Checkpoint directory')
    parser.add_argument('--weights-dir', type=str, default='./Mypackage/resources/weights', help='Weights directory')
    parser.add_argument('--pretrained-path', type=str, default='./Mypackage/resources/ViT-B-32.pt', help='Pretrained model path')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = BoneMorphoNetTrainer(args)
    
    # Check if training should be resumed
    if args.resume and args.checkpoint_path:
        trainer.load_checkpoint(args.checkpoint_path)
    
    # Start training
    trainer.train()

if __name__ == '__main__':
    main()