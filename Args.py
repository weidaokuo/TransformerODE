from dataclasses import dataclass, field
from typing import Tuple, Optional, Literal
import os
from typing import ClassVar

# 
current_dir = os.path.dirname(os.path.abspath(__file__))


# print(current_dir)
# print(current_dir + "/data/origin_data/train_curvle")

@dataclass
class ModelArgs:

    element_list = ["Nb", "W", "Mo", "Zr", "C", "Hf", "Ti", "Ta", "Si", "N", "V", "Sc", "Ru", "B", "Y"]
    processing_list = ["Anneal_Temp_1", "Anneal_Time_1/h", "Quench_1", "Anneal_Temp_2", "Anneal_Time_2/h"]
    extract_Paper_Num = [13]
    alloy_numbers = [ 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
                      67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 84, 85, 86, 87, 90,
                      91, 92, 95, 96, 97, 98, 99
                    ]

    origin_curve_path = './data/origin_curve_data'
    strain_stress_curve_save_path = './data/process_curve_data'
    excell_data_path = './data/data_all.xlsx'

    save_origin_data = True

    physic_names = [
            'Configuration entropy',
            'Mixing enthalpy',
            'Yang delta',
            'Electronegativity delta',
            'VEC mean',
            'Yang omega',
            'Mean cohesive energy',
            'Shear modulus mean',
            'Atomic weight mean',
            'MagpieData mean MeltingT',
            'Interant d electrons',
            'MagpieData mean CovalentRadius',
            'MagpieData mean Electronegativity'
        ]

    state_dim = 3
    cond_dim = 18
    hidden_dim = 128
    batch_size = 1
    epoch = 10000
    max_strain = 33.5
    dropout = 0.1
    num_heads = 4
    num_layers = 2
    inference_n = 10
    
    noise_dim = 4
    num_experts = 5
    
    
    
    
args = ModelArgs()