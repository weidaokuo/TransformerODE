
from pathlib import Path

HERE = Path(__file__).resolve().parent         
WDK_ROOT = HERE.parent                          
DATA_DIR = WDK_ROOT / 'data'
RESULTS_DIR = HERE / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

PROCESSED_JSONL = DATA_DIR / 'processed_data.jsonl'
CSV_DATA = DATA_DIR / 'csv_data.csv'
FEATURE_STATS = DATA_DIR / 'feature_stats.txt'
STRESS_MEAN_STD = DATA_DIR / 'stress_mean_std.json'

TARGET_ALLOYS = [47, 48, 49, 50, 51, 52, 81, 84, 85, 86, 95, 96, 97, 98]
TARGET_PROCESS = [1450.0, 1.0, 0.0, 25.0, 0.0]  

ELEMENT_LIST = ["Nb", "W", "Mo", "Zr", "C", "Hf", "Ti", "Ta",
                "Si", "N", "V", "Sc", "Ru", "B", "Y"]

PHYSIC_NAMES = [
    'Configuration entropy', 'Mixing enthalpy', 'Yang delta',
    'Electronegativity delta', 'VEC mean', 'Yang omega',
    'Mean cohesive energy', 'Shear modulus mean', 'Atomic weight mean',
    'MagpieData mean MeltingT', 'Interant d electrons',
    'MagpieData mean CovalentRadius', 'MagpieData mean Electronegativity'
]
PROCESS_NAMES = ["Anneal_Temp_1", "Anneal_Time_1/h", "Quench_1",
                 "Anneal_Temp_2", "Anneal_Time_2/h"]


ALPHA_TAYLOR = 0.3        
BURGERS_B = 2.86e-10        
SIGMA_I = 0.0               


R2_GOOD = 0.985            
MAX_STRAIN_CAP = 30.0       
SAT_FRACTION = 0.99       


N_SYNTHETIC = 100          
SEED = 42


ODE_COND_DIM = 18
ODE_COND_DIM_V2 = 21    
ODE_HIDDEN_DIM = 128
ODE_NUM_HEADS = 4
ODE_NUM_LAYERS = 2
ODE_NOISE_DIM = 16
ODE_NUM_EXPERTS = 5
ODE_LR = 7e-4
ODE_WEIGHT_DECAY = 1e-5
ODE_ETA_MIN = 1e-6
ODE_GRAD_CLIP = 0.5
ODE_N_EPOCHS = 5000
ODE_CKPT_EVERY = 100


def parse_feature_stats(path=FEATURE_STATS):
    names, means, stds = [], [], []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(('=', '-', 'Feature-wise', 'Feature Name', 'Note:')):
                continue
            parts = line.rsplit(None, 2)   
            if len(parts) != 3:
                continue
            try:
                means.append(float(parts[1]))
                stds.append(float(parts[2]))
                names.append(parts[0])
            except ValueError:
                continue
    return names, means, stds
