import pandas as pd
from pymatgen.core import Composition
from matminer.featurizers.composition import WenAlloys, ElementProperty
from matminer.featurizers.base import MultipleFeaturizer
import numpy as np
import warnings
from Args import args

warnings.filterwarnings("ignore", category=FutureWarning, module="matminer")


class ObtainFeatures:
    def __init__(self, element_list, composition_list):
        filtered = [(elem, comp) for elem, comp in zip(element_list, composition_list) if comp > 1e-8]
        if not filtered:
            raise ValueError("所有成分均为零，无法构建合金。")

        elems, comps = zip(*filtered)

        if max(comps) > 1.0:
            total = sum(comps)
            mole_fractions = [c / total for c in comps]
        else:
            total = sum(comps)
            mole_fractions = [c / total for c in comps]

        formula_str = "".join(f"{elem}{frac:.6g}" for elem, frac in zip(elems, mole_fractions))
        self.composition = Composition(formula_str)

        try:
            self.featurizer = MultipleFeaturizer([
                WenAlloys(),
                ElementProperty.from_preset("magpie")
            ])
        except TypeError as e:
            try:
                print("尝试使用旧版本 matminer API...")
                self.featurizer = MultipleFeaturizer([
                    WenAlloys(impute_nan=True),
                    ElementProperty.from_preset("magpie", impute_nan=True)
                ])
            except TypeError:
                print("尝试基本初始化...")
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
            print(f"提取特征时出错: {e}")
            return pd.Series()

    def get_key_features(self):
        all_feat = self.get_features()
        if all_feat.empty:
            print("警告：特征提取返回空结果")
            return pd.Series()

        key_names = args.physic_names
        available_keys = [k for k in key_names if k in all_feat.index]
        return all_feat[available_keys]


def extract_hea_features_batch(df_compositions, element_list, norm = False):
    if df_compositions.shape[1] != len(element_list):
        raise ValueError(
            f"df_compositions 列数 ({df_compositions.shape[1]}) 与 element_list 长度 ({len(element_list)}) 不匹配！")

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
                print(f"第 {idx} 行特征提取返回空结果")
        except Exception as e:
            print(f"警告：第 {idx} 行特征提取失败，跳过。错误: {e}")

    if not feature_list:
        raise RuntimeError("没有成功提取任何样本的特征！")

    features_raw = pd.concat(feature_list, axis=1).T
    features_raw.index = valid_indices  

    expected_cols = args.physic_names

    for col in expected_cols:
        if col not in features_raw.columns:
            print(f"警告：预期特征 '{col}' 不存在，用 NaN 填充")
            features_raw[col] = np.nan


    features_raw = features_raw[expected_cols]
    if norm:
        save_stats_path = "data/feature_stats.txt"

        means = features_raw.mean()
        stds = features_raw.std()


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
                print(f"注意：特征 '{col}' 的标准差为 {std_val}，无法标准化，保留原始值。")
                features_scaled[col] = features_raw[col]
            else:
                features_scaled[col] = (features_raw[col] - means[col]) / std_val

        return features_raw, features_scaled
    else:
        return features_raw


if __name__ == "__main__":
    element_list = ["Nb", "W", "Mo", "Zr", "C", "Hf", "Ti", "Ta", "Si", "N", "V", "Sc", "Ru", "B", "Y"]


    data = [
        [20, 10, 10, 50, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [10, 20, 20, 40, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [40, 10, 10, 30, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]

    df_comp = pd.DataFrame(data, columns=element_list)
    print("输入数据:")
    print(df_comp)
    print("\n" + "=" * 80 + "\n")

    raw_feat = extract_hea_features_batch(
        df_compositions=df_comp,
        element_list=element_list,
        norm=False
    )
    print("\n原始特征:")
    print(raw_feat)


