from data_aug import get_data_augmentations
from dataset import load_train_data, load_val_data
from engine import train, validate
from log import create_logger
from math import ceil
from os import get_terminal_size
from scaler import NativeScaler
from throughput import throughput
from timm.loss.cross_entropy import SoftTargetCrossEntropy
from timm.layers import RmsNorm
from timm.models import create_model

import argparse, sys, torch
import torch.nn as nn
import torch.optim as optim
from functools import partial

MODEL_CONFIGS = {
    'cait': {
        'name': 'cait_s36_384',
        'default_batch_size': 128,
        'paper_acc1': 86.74,
        'weights': 'https://github.com/ehuynh1106/TinyImageNet-Transformers/releases/download/weights/cait_s36_384.pth',
    },
    'deit': {
        'name': 'deit_base_distilled_patch16_384',
        'default_batch_size': 64,
        'paper_acc1': 87.29,
        'weights': 'https://github.com/ehuynh1106/TinyImageNet-Transformers/releases/download/weights/deit_base_distilled_384.pth',
    },
    'swin': {
        'name': 'swin_large_patch4_window12_384',
        'default_batch_size': 32,
        'paper_acc1': 91.35,
        'weights': 'https://github.com/ehuynh1106/TinyImageNet-Transformers/releases/download/weights/swin_large_384.pth',
    },
    'vit': {
        'name': 'vit_large_patch16_384',
        'default_batch_size': 64,
        'paper_acc1': 86.43,
        'weights': 'https://github.com/ehuynh1106/TinyImageNet-Transformers/releases/download/weights/vit_large_384.pth',
    },
}

def parse_args():
    parser = argparse.ArgumentParser('Vision Transformer training and evaluation script', add_help=False)
    parser.add_argument('--model', type=str, required=True, choices=list(MODEL_CONFIGS.keys()),
                        help='vit: ViT-L/16, '
                             'deit: DeiT-B/16 distilled, '
                             'swin: Swin-L, '
                             'cait: CaiT-S36')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true', help='Train a model')
    group.add_argument('--evaluate', nargs='?', const='paper', help='Evaluate a trained model. Use no value for paper weights.')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--resume', type=str, help='Resume training')
    parser.add_argument('--data-root', type=str, default='.', help='Directory containing train_dataset.pkl and val_dataset.pkl')
    parser.add_argument('--batch-size', type=int, default=None, help='Override the default batch size')
    parser.add_argument('--eval-batch-size', type=int, default=None, help='Override validation batch size')
    parser.add_argument('--max-eval-samples', type=int, default=None,
                        help='Evaluate only the first N validation samples for smoke tests')
    parser.add_argument('--amp-eval', action='store_true', help='Use autocast mixed precision during evaluation')
    parser.add_argument('--img-size', type=int, default=384, help='Input image size')
    parser.add_argument('--norm-layer', type=str, default='layernorm', choices=['layernorm', 'rmsnorm'],
                        help='Transformer normalization layer')
    parser.add_argument('--rmsnorm-eps', type=float, default=1e-6, help='Epsilon for RMSNorm')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--true-batch-size', type=int, default=128, help='Effective batch size for gradient accumulation')
    parser.add_argument('--num-workers', type=int, default=0, help='DataLoader workers')
    parser.add_argument('--pretrained', action='store_true', default=True, help='Initialize from timm pretrained weights for training')
    parser.add_argument('--no-pretrained', action='store_false', dest='pretrained', help='Do not initialize from timm pretrained weights')

    # data augmentation
    parser.add_argument('--mixup', action='store_true', default=True)
    parser.add_argument('--no-mixup', action='store_false', dest='mixup', help='Disable mixup')
    parser.add_argument('--cutmix', action='store_true', default=True)
    parser.add_argument('--no-cutmix', action='store_false', dest='cutmix', help='Disable cutmix')
    parser.add_argument('--randerase', action='store_true', default=True)
    parser.add_argument('--no-randerase', action='store_false', dest='randerase', help='Disable random erasing')
    parser.add_argument('--randaug', action='store_true', default=False)

    #optimizer
    parser.add_argument('--optim', type=str, default='adamw', choices=['adamw', 'sgd'])
    parser.add_argument('--nesterov', action='store_true', default=True, help='Use nesterov momentum for SGD')
    parser.add_argument('--no-nesterov', action='store_false', dest='nesterov', help='Disable nesterov')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for optimizer')
    parser.add_argument('--wd', type=float, default=0.05, help='Weight decay for optimizer')

    parser.add_argument('--label-smooth', type=float, default=0.1, help='Label smoothing percent')

    args, _ = parser.parse_known_args()

    return args

def resolve_checkpoint(model_name, evaluate):
    if not evaluate:
        return None
    if evaluate == 'paper':
        return MODEL_CONFIGS[model_name]['weights']
    return evaluate

def load_checkpoint(path, device):
    if path.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(path, map_location=device)
    else:
        checkpoint = torch.load(path, map_location=device)
    return checkpoint

def checkpoint_state_dict(checkpoint):
    for key in ('model_state_dict', 'state_dict', 'model'):
        if isinstance(checkpoint, dict) and key in checkpoint:
            return checkpoint[key]
    return checkpoint

def build_norm_layer(norm_layer, rmsnorm_eps):
    if norm_layer == 'rmsnorm':
        return partial(RmsNorm, eps=rmsnorm_eps)
    return None

def load_model(model_name, evaluate, batch_size_override=None, pretrained=True, device='cpu', norm_layer=None):
    cfg = MODEL_CONFIGS[model_name]
    checkpoint_path = resolve_checkpoint(model_name, evaluate)
    model_kwargs = {
        'pretrained': pretrained and checkpoint_path is None,
        'drop_path_rate': 0.1,
    }
    if norm_layer is not None:
        model_kwargs['norm_layer'] = norm_layer
    model = create_model(
        cfg['name'],
        **model_kwargs,
    )
    batch_size = batch_size_override or cfg['default_batch_size']

    for param in model.parameters():
        param.requires_grad = False
    model.reset_classifier(num_classes=200)

    if checkpoint_path:
        checkpoint = load_checkpoint(checkpoint_path, device)
        state_dict = checkpoint_state_dict(checkpoint)
        model.load_state_dict(state_dict)

    return model, batch_size

def load_optimizer(args, model):
    if args.optim == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9, nesterov=args.nesterov)
    else:
        logger.error('Invalid optimizer name, please use either adamw or sgd')
        sys.exit(1)

    return optimizer

if __name__ == '__main__':
    logger = create_logger()
    args = parse_args()

    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.norm_layer != 'layernorm' and args.evaluate == 'paper':
        raise ValueError('Paper checkpoints were trained with LayerNorm. Use --norm-layer layernorm for paper reproduction.')
    if args.norm_layer != 'layernorm' and args.pretrained and not args.evaluate:
        logger.warning('RMSNorm changes the architecture, so timm LayerNorm pretrained initialization is disabled.')
        args.pretrained = False
    norm_layer = build_norm_layer(args.norm_layer, args.rmsnorm_eps)
    model, batch_size = load_model(
        args.model,
        args.evaluate,
        batch_size_override=args.batch_size,
        pretrained=args.pretrained,
        device=device,
        norm_layer=norm_layer,
    )
    model = model.to(device)
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    if args.evaluate:
        logger.info(f"Paper Acc@1 for {args.model}: {MODEL_CONFIGS[args.model]['paper_acc1']}")

    update_freq = max(1, args.true_batch_size // batch_size)
    val_batch_size = args.eval_batch_size or (batch_size if not args.throughput else 32)
    val_loader = load_val_data(
        args.img_size,
        val_batch_size,
        args.data_root,
        args.num_workers,
        args.max_eval_samples,
    )
    
    if args.throughput:
        logger.info(f"Testing throughput of {args.model}")
        throughput(val_loader, model, logger)
    elif args.evaluate:
        logger.info(f"Evaluating {args.model} on {len(val_loader.dataset)} validation samples")
        val_loss, top_1_val_acc, top_5_val_acc = validate(model, device, val_loader, -1, use_amp=args.amp_eval)
        logger.info(f"Top 1 Validation Accuracy: {top_1_val_acc}\tTop 5 Validation Accuracy: {top_5_val_acc}")
    elif args.train:
        randaug_magnitude = 9 if args.randaug else 0
        start_epoch = 0
        epochs = args.epochs

        train_loader = load_train_data(args.img_size, randaug_magnitude, batch_size, args.data_root, args.num_workers)

        mixup, random_erase = get_data_augmentations(
            args.label_smooth,
            en_mixup=args.mixup,
            en_cutmix=args.cutmix,
            en_randerase=args.randerase
        )
    
        if mixup:
            loss_fn = SoftTargetCrossEntropy()
        else:
            loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)
        
        optimizer = load_optimizer(args, model)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ceil(len(train_loader)/update_freq)*epochs)
        loss_scaler = NativeScaler()

        if args.resume:
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['start_epoch'] + 1
            model.load_state_dict(checkpoint['model_state_dict'])

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            loss_scaler.load_state_dict(checkpoint['loss_scaler_state_dict'])
            del checkpoint

        logger.info(f"Start training {args.model} at epoch {start_epoch + 1}")
        logger.info(f"Current lr: {optimizer.param_groups[0]['lr']}")

        for i in range(start_epoch, epochs):
            logger.info(f"Epoch {i+1}")
            train_loss = train(model, loss_fn, optimizer, device, train_loader, scheduler, loss_scaler, update_freq, mixup, random_erase)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss_scaler_state_dict': loss_scaler.state_dict(),
                'start_epoch': i,
            }, f'models/epoch{i+1}.pth')
            val_loss, top_1_val_acc, top_5_val_acc = validate(
                model,
                device,
                val_loader,
                i,
                can_visualize=i>=epochs//2,
                use_amp=args.amp_eval,
            )

            logger.info(f"Current lr: {optimizer.param_groups[0]['lr']}")
            logger.info(f"Top 1 Validation Accuracy: {top_1_val_acc}\tTop 5 Validation Accuracy: {top_5_val_acc}")
            try:
                width = get_terminal_size().columns
            except OSError:
                width = 80
            print("-"*width)
