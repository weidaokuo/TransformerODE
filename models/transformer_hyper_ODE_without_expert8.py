import torch
import torch.nn as nn
import torchdiffeq
import math
import numpy as np


class RotaryPositionalEncoding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

    def forward(self, x, t):
        if t.dim() == 1:
            t = t.unsqueeze(0).expand(x.size(0), -1)
            
        t_expanded = t.unsqueeze(-1)
        inv_freq_expanded = self.inv_freq.unsqueeze(0).unsqueeze(0).to(t.device)
        
        freqs = torch.matmul(t_expanded, inv_freq_expanded)
        
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        
        cos_freq = torch.cos(freqs)
        sin_freq = torch.sin(freqs)
        
        out = torch.zeros_like(x)
        out[..., 0::2] = x1 * cos_freq - x2 * sin_freq
        out[..., 1::2] = x1 * sin_freq + x2 * cos_freq
        
        return out


class LatentVariableTransformer(nn.Module):
    def __init__(self, hidden_dim=32, num_heads=4, num_layers=2, cond_dim=18, noise_dim=4, num_experts=5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim
        self.num_experts = num_experts
        input_dim = 1 + noise_dim 
        self.input_proj = nn.Linear(input_dim, hidden_dim)       
        self.memory_proj = nn.Linear(cond_dim, hidden_dim)        
        self.rope = RotaryPositionalEncoding(hidden_dim)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=0.1,
            activation='gelu', batch_first=True
        )
        
        self.transformer_decoder = nn.ModuleList([
            nn.TransformerDecoder(decoder_layer, num_layers=num_layers) 
            for _ in range(num_experts)
        ])
               
        self.out_proj = nn.Linear(hidden_dim, 1)

    def forward(self, strain_seq, times, conditions, latent_noise=None, expert_id=None):
        B, T, _ = strain_seq.shape
        device = strain_seq.device        
        memory = self.memory_proj(conditions).unsqueeze(1)
        if latent_noise is None:
            latent_noise = torch.randn(B, self.noise_dim, device=device)        
        latent_noise_expanded = latent_noise.unsqueeze(1).expand(-1, T, -1)        
        target_concat = torch.cat([
            strain_seq,           
            latent_noise_expanded 
        ], dim=-1)
        

        x = self.input_proj(target_concat)
        x = self.rope(x, times)
        
        # Select expert and perform Transformer computation 

        # Logic:
        # 1. If expert_id is specified, use that ID.
        # 2. If in training mode (self.training), randomly select an expert.
        # 3. If in inference mode and no ID is specified, default to expert 0
        #    (or alternatively, select randomly).
        
        if expert_id is None:
            if self.training:
                selected_expert_id = np.random.randint(0, self.num_experts)
            else:
                selected_expert_id = 0
        else:
            selected_expert_id = expert_id
            
        current_decoder = self.transformer_decoder[selected_expert_id]
        causal_mask = self.generate_causal_mask(T, device)
        output = current_decoder(tgt=x, memory=memory, tgt_mask=causal_mask)
        
        return self.out_proj(output)
        

    def generate_causal_mask(self, seq_len, device):
        """
        Generate a causal mask to prevent the Transformer from attending to future information.

        Args:
            seq_len (int): Sequence length T
            device: Device (cuda or cpu)

        Returns:
            torch.Tensor: Mask matrix of shape [T, T]
        """
        
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


class LatentODEVectorField(nn.Module):
    def __init__(self, transformer_model, strain_grid, conditions, latent_noise, expert_id=None):
        super().__init__()
        self.transformer = transformer_model
        self.strain_grid = strain_grid
        self.conditions = conditions
        self.latent_noise = latent_noise 
        self.expert_id = expert_id
        self._precomputed_rates = None
        self._compute_rates()

    def _compute_rates(self):
        B, T = self.strain_grid.shape
        device = self.strain_grid.device        
        strain_input = self.strain_grid.unsqueeze(-1)
        
        self._precomputed_rates = self.transformer(
            strain_input, 
            self.strain_grid, 
            self.conditions,
            latent_noise=self.latent_noise,
            expert_id = self.expert_id
        )

    def forward(self, t, z):
        B = z.size(0)
        device = z.device
        
        if t.numel() == 1:
            t_expanded = t.unsqueeze(0).expand(B, 1).contiguous()
        else:
            t_expanded = t.contiguous()

        ref_grid = self.strain_grid[0]
        idx_right = torch.searchsorted(ref_grid, t_expanded)
        idx_left = (idx_right - 1).clamp(0, self.strain_grid.size(1) - 2)
        idx_right_clamped = (idx_right).clamp(1, self.strain_grid.size(1) - 1)

        t_left = torch.gather(self.strain_grid, 1, idx_left)
        t_right = torch.gather(self.strain_grid, 1, idx_right_clamped)

        rate_left = torch.gather(self._precomputed_rates, 1, idx_left.unsqueeze(-1).expand(-1, -1, 1)).squeeze(-1)
        rate_right = torch.gather(self._precomputed_rates, 1, idx_right_clamped.unsqueeze(-1).expand(-1, -1, 1)).squeeze(-1)

        dt = (t_right - t_left) + 1e-8
        alpha = (t_expanded - t_left) / dt
        
        rate_at_t = rate_left + alpha * (rate_right - rate_left)
        
        return rate_at_t



class LatentVariableNeuralODE(nn.Module):
    def __init__(self, cond_dim=18, hidden_dim=32, num_heads=4, num_layers=2, noise_dim=4, num_experts=5):
        super().__init__()
        self.transformer = LatentVariableTransformer(hidden_dim, num_heads, num_layers, cond_dim, noise_dim, num_experts)
        self.initial_stress_bias = nn.Parameter(torch.tensor([0.0]))

    def forward(self, times, conditions, latent_noise=None, expert_id=None):
        B = conditions.size(0)
        device = conditions.device
        
        if times.dim() == 1:
            times_flat = times.unsqueeze(0).expand(B, -1)
        else:
            times_flat = times
            
        ode_func = LatentODEVectorField(self.transformer, times_flat, conditions, latent_noise, expert_id)
        
        z0 = self.initial_stress_bias.expand(B, 1)
        
        t_eval = times_flat[0]
        
        z_traj = torchdiffeq.odeint(
            func=ode_func, 
            y0=z0, 
            t=t_eval, 
            method='dopri5',
            rtol=1e-3,                  
            atol=1e-4
        )
        
        z_traj = z_traj.permute(1, 0, 2)
        
        return {
            'stress': z_traj,
            'hardening_rate': ode_func._precomputed_rates 
        }
        
        
        
        
        
        