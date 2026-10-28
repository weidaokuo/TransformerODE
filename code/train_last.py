
import json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

WDK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WDK_ROOT)
sys.path.insert(0, os.path.join(WDK_ROOT, 'mac_test'))

from models.transformer_hyper_ODE_without_expert8 import LatentVariableNeuralODE
from Args import args
from data_loader import StressStrainDataset

import warnings
warnings.filterwarnings('ignore')

SEED = 42
import random
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

N_EPOCHS = 10000
LR = 7e-4

jsonl_file = sys.argv[1]
mean_std_file = sys.argv[2]
SAVE_DIR = sys.argv[3]
os.makedirs(SAVE_DIR, exist_ok=True)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print(f"训练: {jsonl_file} → {SAVE_DIR} ({N_EPOCHS} epochs, lr={LR})")

dataset = StressStrainDataset(jsonl_path=jsonl_file, mean_std_path=mean_std_file)
print(f"Total samples: {len(dataset)}")

train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, collate_fn=None)

model = LatentVariableNeuralODE(
    cond_dim=args.cond_dim, hidden_dim=args.hidden_dim,
    num_heads=args.num_heads, num_layers=args.num_layers,
    num_experts=args.num_experts, noise_dim=args.noise_dim
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-6)
model.train()

loss_log_path = os.path.join(SAVE_DIR, 'loss_log.txt')
global_step = 0
start_epoch = 0


latest_ckpt = None
latest_epoch = 0
if os.path.exists(SAVE_DIR):
    for file in os.listdir(SAVE_DIR):
        if file.startswith('checkpoint_epoch_') and file.endswith('.pth'):
            try:
                epoch_num = int(file.replace('checkpoint_epoch_', '').replace('.pth', ''))
                if epoch_num > latest_epoch:
                    latest_epoch = epoch_num
                    latest_ckpt = os.path.join(SAVE_DIR, file)
            except:
                pass
if latest_ckpt is not None:
    print(f"发现checkpoint: {latest_ckpt}, 从epoch {latest_epoch}继续")
    checkpoint = torch.load(latest_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    global_step = checkpoint.get('global_step', start_epoch * len(dataset))
else:
    print("从头开始训练")

start_time = time.time()
for i in range(start_epoch, N_EPOCHS):
    epoch_loss_sum = 0.0
    batch_count = 0
    for batch_idx, batch in enumerate(train_dataloader):
        conditions = batch['initial_feature'].to(device)
        strain_raw = batch['strain'].squeeze(dim=0).to(device)
        stress_raw = batch['stress'].to(device)

        strain_normalized = strain_raw / args.max_strain
        B = conditions.size(0)
        times = strain_normalized.unsqueeze(0).expand(B, -1)

        try:
            outputs = model(times, conditions)
            stress_pred = outputs['stress']
            if stress_pred.dim() == 3:
                stress_pred = stress_pred.squeeze(-1)
        except Exception as e:
            print(f"Forward Error at Epoch {i}, Batch {batch_idx}: {e}")
            continue

        loss = F.mse_loss(stress_pred, stress_raw)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Loss NaN/Inf at Epoch {i}, Batch {batch_idx}. Skipping.")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss_sum += loss.item()
        batch_count += 1
        global_step += 1

    if batch_count == 0:
        print(f"Epoch {i}: No valid batches.")
        continue

    loss_mean = epoch_loss_sum / batch_count
    scheduler.step()

    elapsed = time.time() - start_time
    avg_time_per_epoch = elapsed / max(i - start_epoch + 1, 1)
    eta = avg_time_per_epoch * (N_EPOCHS - i - 1)

    if (i + 1) % 1000 == 0 or (i + 1) == N_EPOCHS:
        ckpt_path = os.path.join(SAVE_DIR, f'checkpoint_epoch_{i+1}.pth')
        torch.save({
            'epoch': i + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss_mean,
            'global_step': global_step,
            'args': args
        }, ckpt_path)
        print(f"  Checkpoint已保存: {ckpt_path}")

    with open(loss_log_path, 'a') as f:
        f.write(f"{i+1},{loss_mean}\n")

    if (i + 1) % 500 == 0 or i < 10:
        print(f"Epoch [{i+1:5d}/{N_EPOCHS}] | Loss: {loss_mean:.6f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e} | "
              f"Time: {avg_time_per_epoch:.2f}s/epoch | ETA: {eta:.0f}s")

print(f"训练完成！总训练轮数: {N_EPOCHS - start_epoch}")
