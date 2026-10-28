import pandas as pd
from pymatgen.core import Composition
from matminer.featurizers.composition import WenAlloys, ElementProperty
from matminer.featurizers.base import MultipleFeaturizer
import numpy as np
import warnings
from Args import args

warnings.filterwarnings("ignore", category=FutureWarning, module="matminer")


class ObtainFeatures:
    """
    High-Entropy Alloy Feature Extractor (Single Sample)
    """

    def __init__(self, element_list, composition_list):
        filtered = [(elem, comp) for elem, comp in zip(element_list, composition_list) if comp > 1e-8]
        if not filtered:
            raise ValueError("All compositions are zero; cannot construct alloy.")

        elems, comps = zip(*filtered)

        # Automatically determine whether it is at%.
        if max(comps) > 1.0:
            total = sum(comps)
            mole_fractions = [c / total for c in comps]
        else:
            total = sum(comps)
            mole_fractions = [c / total for c in comps]

        formula_str = "".join(f"{elem}{frac:.6g}" for elem, frac in zip(elems, mole_fractions))
        self.composition = Composition(formula_str)

        # Trying different versions of the matminer API...
        try:
            self.featurizer = MultipleFeaturizer([
                WenAlloys(),
                ElementProperty.from_preset("magpie")
            ])
        except TypeError as e:
            try:
                print("Attempting to use the older matminer API...")
                self.featurizer = MultipleFeaturizer([
                    WenAlloys(impute_nan=True),
                    ElementProperty.from_preset("magpie", impute_nan=True)
                ])
            except TypeError:
                print("Attempting basic initialization...")
                self.featurizer = MultipleFeaturizer([
                    WenAlloys(),
                    ElementProperty.from_preset("magpie")
                ])

        self.feature_labels = self.featurizer.feature_labels()

    def get_features(self, as_dataframe=False):
        try:
            features = self.featurizer.featurize(self.composition)
            series = pd.Series(features, index=self.feature_labels)
            return series.to_frame().T if as_dataframe else series
        except Exception as e:
            print(f"Error extracting features: {e}")
            return pd.Series()

    def get_key_features(self):
        all_feat = self.get_features()
        if all_feat.empty:
            print("Warning: feature extraction returned an empty result.")
            return pd.Series()

        key_names = args.physic_names
        available_keys = [k for k in key_names if k in all_feat.index]
        return all_feat[available_keys]


def extract_hea_features_batch(df_compositions, element_list, norm = False):
    """
    Batch-extract key high-entropy alloy features and standardize them (z-score). 
    Cases with std=0 are handled automatically.
    """
    if df_compositions.shape[1] != len(element_list):
        raise ValueError(
            f"Number of columns in df_compositions ({df_compositions.shape[1]}) does not match length of element_list ({len(element_list)})!")

    df_compositions = df_compositions.copy()
    df_compositions.columns = element_list

    feature_list = []
    valid_indices = []
    for idx, row in df_compositions.iterrows():
        try:
            extractor = ObtainFeatures(element_list, row.tolist())
            key_feat = extractor.get_key_features()
            if not key_feat.empty:
                key_feat.name = idx
                feature_list.append(key_feat)
                valid_indices.append(idx)
            else:
                print(f"Feature extraction for row {idx} returned an empty result.")
        except Exception as e:
            print(f"Warning: feature extraction failed for row {idx}; skipping. Error: {e}")

    if not feature_list:
        raise RuntimeError("Failed to extract features for any sample!")

    features_raw = pd.concat(feature_list, axis=1).T
    features_raw.index = valid_indices  

    expected_cols = args.physic_names

    for col in expected_cols:
        if col not in features_raw.columns:
            print(f"Warning: expected feature '{col}' not found; filling with NaN.")
            features_raw[col] = np.nan

    features_raw = features_raw[expected_cols]
    if norm:
        save_stats_path = "data/feature_stats.txt"
        means = features_raw.mean()
        stds = features_raw.std()

        # save
        with open(save_stats_path, 'w', encoding='utf-8') as f:
            f.write("Feature\tMean\tStd\n")
            for col in features_raw.columns:
                mean_val = means[col]
                std_val = stds[col]
                f.write(f"{col}\t{mean_val:.6f}\t{std_val:.6f}\n")
        features_scaled = features_raw.copy()
        for col in features_raw.columns:
            std_val = stds[col]
            if pd.isna(std_val) or std_val == 0:
                print(f"Note: feature '{col}' has a standard deviation of {std_val} and cannot be standardized; original values retained.")
                features_scaled[col] = features_raw[col]
            else:
                features_scaled[col] = (features_raw[col] - means[col]) / std_val

        return features_raw, features_scaled
    else:
        return features_raw



# example
if __name__ == "__main__":
    element_list = ["Nb", "W", "Mo", "Zr", "C", "Hf", "Ti", "Ta", "Si", "N", "V", "Sc", "Ru", "B", "Y"]

    # Construct an example DataFrame (3 alloy samples)
    data = [
        [20, 10, 10, 50, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [10, 20, 20, 40, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [40, 10, 10, 30, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]

    df_comp = pd.DataFrame(data, columns=element_list)
    print("input data:")
    print(df_comp)
    print("\n" + "=" * 80 + "\n")
    
    raw_feat = extract_hea_features_batch(
        df_compositions=df_comp,
        element_list=element_list,
        norm=False
    )
    print("\n origin feature:")
    print(raw_feat)


