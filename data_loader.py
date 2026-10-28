import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json


def custom_collate_fn(batch):
    paper_nums = [item['Paper_Num'] for item in batch]
    alloy_nums = [item['Alloy_Num'] for item in batch]
    initial_features = torch.stack([item['initial_feature'] for item in batch], dim=0)
    strains = [item['strain'] for item in batch]  
    stresses = [item['stress'] for item in batch] 

    return {
        'Paper_Num': paper_nums,
        'Alloy_Num': alloy_nums,
        'initial_feature': initial_features,
        'strain': strains,  
        'stress': stresses  
    }


class StressStrainDataset(Dataset):
    def __init__(self, jsonl_path: str, mean_std_path: str):
        """
        :param jsonl_path: .jsonl path
        :param mean_std_path: contain global_mean_stress and global_std_stress path
        """
        self.samples = []

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line.strip()))

        with open(mean_std_path, 'r', encoding='utf-8') as f:
            stats = json.load(f)
            self.global_mean_stress = stats['global_mean_stress']
            self.global_std_stress = stats['global_std_stress']

        self.global_mean_stress = float(self.global_mean_stress)
        self.global_std_stress = float(self.global_std_stress)

        assert self.global_std_stress > 0, "Standard deviation must be positive!"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        paper_num = sample['Paper_Num']
        alloy_num = sample['Alloy_Num']
        initial_feature = torch.tensor(sample['initial_feature'], dtype=torch.float32)

        curve = torch.tensor(sample['strain_stress_curve'], dtype=torch.float32)  # shape: [N, 2]
        strain = curve[:, 0]  
        stress = curve[:, 1]  

        stress_normalized = (stress - self.global_mean_stress) / self.global_std_stress

        return {
            'Paper_Num': paper_num,
            'Alloy_Num': alloy_num,
            'initial_feature': initial_feature,
            'strain': strain,
            'stress': stress_normalized
        }


# example
if __name__ == '__main__':
    
    jsonl_file = './data/output_interpolated_stress.jsonl'
    mean_std_file = './data/stress_mean_std.json'

    # create dataset 
    dataset = StressStrainDataset(jsonl_path=jsonl_file, mean_std_path=mean_std_file)
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,  
        collate_fn=None  
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=custom_collate_fn
    )

    # test
    for i, batch in enumerate(dataloader):
        print(f"Batch {i + 1}:")
        print("Paper_Nums:", batch['Paper_Num'])
        print("Initial feature shape:", batch['initial_feature'].shape)
        print("Strain lengths:", [len(s) for s in batch['strain']])
        print("Stress (first sample, first 5 values):", batch['stress'][0][:5])
        if i == 0:
            break