
import os, sys, json, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

WDK_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  
sys.path.insert(0, WDK_ROOT)
sys.path.insert(0, os.path.join(WDK_ROOT, 'voce_km_latent'))
sys.path.insert(0, os.path.join(WDK_ROOT, 'mac_test'))

from models.transformer_hyper_ODE_latent3 import LatentVariableNeuralODE
from config import (ODE_COND_DIM_V2, ODE_HIDDEN_DIM, ODE_NUM_HEADS, ODE_NUM_LAYERS,
                    ODE_NOISE_DIM, ODE_NUM_EXPERTS, ODE_LR, ODE_WEIGHT_DECAY,
                    ODE_ETA_MIN, ODE_GRAD_CLIP)

MAX_STRAIN_REF = 33.5  

QUANT_NAMES = ['rho', 'theta', 'Lambda', 'sigma']
LOSS_COLS = [0, 1, 3]  

N_EPOCHS = 600
SEED = 20260830
EPS_TRUE_MAX = 0.55  

class LatentDataset(Dataset):
    """任务一同口径: latent_curve = [strain_pct, rho, theta, Lambda, sigma] (物理值)"""
    def __init__(self, jsonl_path, zstats):
        self.rows = []
        with open(jsonl_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    lc = np.array(r['latent_curve'], dtype=float)  
                    self.rows.append({
                        'cond': torch.tensor(r['initial_feature'], dtype=torch.float32),
                        'strain': torch.tensor(lc[:, 0], dtype=torch.float32),
                        'latent': torch.tensor(lc[:, 1:], dtype=torch.float32),  
                    })
        self.zstats = zstats

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        s = self.rows[idx]
        z = torch.zeros_like(s['latent'])
        for d, q in enumerate(QUANT_NAMES):
            z[:, d] = (s['latent'][:, d] - self.zstats[q]['mean']) / self.zstats[q]['std']
        return {'cond': s['cond'], 'strain': s['strain'], 'latent': z}


def fit_zstats(idx_list, latent_rows):
    allv = np.concatenate([latent_rows[i]['latent'] for i in idx_list], axis=0)  # [N_total,4]
    stats = {}
    for d, q in enumerate(QUANT_NAMES):
        stats[q] = {'mean': float(allv[:, d].mean()), 'std': float(allv[:, d].std())}
    return stats


def load_latent_curves():
    """ latent_curves.jsonl → {idx: {'cond21': np.array[21], 'latent': np.array[T,4]}}"""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'latent_curves.jsonl')
    rows = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                lc = np.array(r['latent_curve'], dtype=float)
                rows[r['Alloy_Num']] = {
                    'cond21': np.array(r['initial_feature'], dtype=float),
                    'latent': lc[:, 1:],  # [T,4] 
                    'strain_pct': lc[:, 0],
                }
    return rows


def collate_latent(batch):
    strain = batch[0]['strain']  # 统一网格
    return {
        'cond': torch.stack([b['cond'] for b in batch]),
        'strain': strain.unsqueeze(0),
        'latent': torch.stack([b['latent'] for b in batch]),
    }


def build_model():
    return LatentVariableNeuralODE(
        cond_dim=ODE_COND_DIM_V2, hidden_dim=ODE_HIDDEN_DIM, num_heads=ODE_NUM_HEADS,
        num_layers=ODE_NUM_LAYERS, num_experts=ODE_NUM_EXPERTS,
        noise_dim=ODE_NOISE_DIM, state_dim=4)


def train_ode(train_idx, zstats, save_dir, n_epochs=N_EPOCHS, seed=SEED, latent_src=None,
              theta_w=1.0, cons_w=0.0, channel_log=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(save_dir, exist_ok=True)

    latent_rows = latent_src if latent_src is not None else load_latent_curves()
    path = os.path.join(save_dir, 'train.jsonl')
    with open(path, 'w') as f:
        for idx in train_idx:
            info = latent_rows[idx]
            lc = info['latent']
            strain_pct = info['strain_pct']
            f.write(json.dumps({'Alloy_Num': int(idx),
                                'initial_feature': info['cond21'].tolist(),
                                'latent_curve': [[float(x) for x in row] for row in
                                                 np.column_stack([strain_pct, lc])]}) + '\n')

    dataset = LatentDataset(path, zstats)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0,
                        collate_fn=collate_latent)
    print(f"  [train_ode] 样本数={len(dataset)}, {n_epochs} epochs, latent3(state_dim=4)")

    device = torch.device('cpu')
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=ODE_LR, weight_decay=ODE_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs,
                                                           eta_min=ODE_ETA_MIN)
    model.train()
    t0 = time.time()
    ch_log = []
    for i in range(n_epochs):
        ep = 0.0
        cnt = 0
        ch = [0.0, 0.0, 0.0]
        for batch in loader:
            cond = batch['cond'].to(device)
            strain = batch['strain'][0].to(device)
            target = batch['latent'][0].to(device)          # [T,4]
            times = (strain / MAX_STRAIN_REF).unsqueeze(0)
            try:
                pred = model(times, cond)['latent'].squeeze(0)   # [T,4]
            except Exception as e:
                print(f"  forward err: {e}")
                continue
            l_rho = F.mse_loss(pred[:, 0], target[:, 0])
            l_th = F.mse_loss(pred[:, 1], target[:, 1])
            l_sig = F.mse_loss(pred[:, 3], target[:, 3])
            loss = l_rho + theta_w * l_th + l_sig
            if cons_w > 0:  
                d_sig = torch.gradient(pred[:, 3], spacing=(strain / MAX_STRAIN_REF,))[0]
                loss = loss + cons_w * F.mse_loss(pred[:, 1], d_sig)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), ODE_GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad()
            if channel_log:
                ch[0] += l_rho.item(); ch[1] += l_th.item(); ch[2] += l_sig.item()
            ep += loss.item()
            cnt += 1
        if cnt == 0:
            continue
        scheduler.step()
        if channel_log:
            ch_log.append([c / cnt for c in ch])
        if (i + 1) % 500 == 0 or i == 0:
            print(f"  ep[{i+1}/{n_epochs}] loss={ep/cnt:.6f} ({time.time()-t0:.0f}s)")
    if i + 1 == 600 and n_epochs > 600:
        torch.save({'model_state_dict': model.state_dict(), 'epoch': i + 1}, os.path.join(save_dir, 'model_600.pth'))
    torch.save({'model_state_dict': model.state_dict(), 'epoch': n_epochs},
               os.path.join(save_dir, 'model.pth'))
    if channel_log:
        json.dump(ch_log, open(os.path.join(save_dir, 'channel_loss.json'), 'w'))
    print(f"  [train_ode] 完成 ({time.time()-t0:.0f}s) → {save_dir}/model.pth")
    return model


def load_ode(save_dir):
    device = torch.device('cpu')
    model = build_model().to(device)
    ckpt = torch.load(os.path.join(save_dir, 'model.pth'), map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.train()
    return model


def predict_det(model, cond21, strain_pct, max_strain=MAX_STRAIN_REF):
    """严格确定性单次前向: eval模式 + noise=0 + expert=0 (无任何随机源)"""
    device = next(model.parameters()).device
    model.eval()
    strain_t = torch.tensor(strain_pct, dtype=torch.float32)
    times = (strain_t / max_strain).unsqueeze(0)
    cond = torch.tensor(cond21, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        out = model(times, cond, torch.zeros(1, ODE_NOISE_DIM), 0)
        z = out['latent'].squeeze(0).numpy()
    model.train()
    return z  


def predict_latent_raw(model, cond21, strain_pct, n_samples=50, max_strain=MAX_STRAIN_REF, seed=None):
    if seed is not None:
        np.random.seed(seed)
    device = next(model.parameters()).device
    strain_t = torch.tensor(strain_pct, dtype=torch.float32)
    times = (strain_t / max_strain).unsqueeze(0)
    cond = torch.tensor(cond21, dtype=torch.float32).unsqueeze(0)
    sl = []
    with torch.no_grad():
        for _ in range(n_samples):
            noise = torch.randn(1, ODE_NOISE_DIM)
            ex = np.random.randint(0, ODE_NUM_EXPERTS)
            out = model(times, cond, noise, ex)
            sl.append(out['latent'].squeeze(0).numpy())
    return np.stack(sl)  # [n, T, 4]


def predict_latent(model, cond21, strain_pct, n_samples=1, max_strain=MAX_STRAIN_REF):
    device = next(model.parameters()).device
    strain_t = torch.tensor(strain_pct, dtype=torch.float32)
    times = (strain_t / max_strain).unsqueeze(0)
    cond = torch.tensor(cond21, dtype=torch.float32).unsqueeze(0)
    if n_samples == 1:
        with torch.no_grad():
            out = model(times, cond, torch.zeros(1, ODE_NOISE_DIM), 0)
            return out['latent'].squeeze(0).numpy()  # [T,4]
    else:
        sl = []
        with torch.no_grad():
            for _ in range(n_samples):
                noise = torch.randn(1, ODE_NOISE_DIM)
                ex = np.random.randint(0, ODE_NUM_EXPERTS)
                out = model(times, cond, noise, ex)
                sl.append(out['latent'].squeeze(0).numpy())
        sa = np.stack(sl)
        return sa.mean(0), sa.std(0)
