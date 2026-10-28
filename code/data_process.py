import pandas as pd
import numpy as np
import os
from obtain_feature import extract_hea_features_batch
from Args import args
import json
import csv

class obtain_data(object):
    def __init__(self, args):
        self.excell_data_path = args.excell_data_path
        self.origin_curve_path = args.origin_curve_path
        self.strain_stress_curve_save_path = args.strain_stress_curve_save_path
        self.data_frame = self._load_data()
        print("self.data_frame:", self.data_frame)
        if args.save_origin_data:
            self.data_frame.to_csv('./data/origin_process_data.csv', index=False, encoding='utf-8')


    def process_data_and_curves(self,
            process_csv_path = "./data/origin_process_data.csv",
            curve_dir = "./data/origin_curve_data",
            output_jsonl_path = "./data/processed_data.jsonl"
    ):

        df = pd.read_csv(process_csv_path)


        feature_cols = args.processing_list + args.physic_names
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺失列: {missing_cols}")

        feature_data = df[feature_cols]
        means = feature_data.mean().values  
        stds = feature_data.std().values 

        stds = np.where(stds == 0, 1.0, stds)

        stats_output_path = "./data/feature_stats.txt"
        os.makedirs(os.path.dirname(stats_output_path), exist_ok=True)
        with open(stats_output_path, 'w', encoding='utf-8') as f_stat:
            f_stat.write("Feature-wise Mean and Standard Deviation (for Z-score normalization)\n")
            f_stat.write("=" * 80 + "\n")
            f_stat.write(f"{'Feature Name':<40} {'Mean':>15} {'Std Dev':>15}\n")
            f_stat.write("-" * 80 + "\n")
            for col, mean_val, std_val in zip(feature_cols, means, stds):
                f_stat.write(f"{col:<40} {mean_val:>15.6f} {std_val:>15.6f}\n")
            f_stat.write("\nNote: Used for standardizing the 18-dimensional initial features.\n")


        os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
        with open(output_jsonl_path, 'w', encoding='utf-8') as fout:
            for idx, row in df.iterrows():
                paper_num = int(row["Paper_Num"])
                alloy_num = int(row["Alloy_Num"])

                curve_filename = f"{paper_num}.{alloy_num}.csv"
                curve_path = os.path.join(curve_dir, curve_filename)
                strain_stress = []
                try:
                    if os.path.exists(curve_path):
                        curve_df = pd.read_csv(curve_path, header=None)
                        if curve_df.shape[1] < 2:
                            print(f"警告: {curve_filename} 列数不足，跳过")
                            continue
                        curve_df = curve_df.iloc[:, :2].apply(pd.to_numeric, errors='coerce')
                        curve_df = curve_df.dropna()
                        strain_stress = curve_df.values.tolist()
                    else:
                        print(f"警告: 曲线文件不存在: {curve_path}")
                except Exception as e:
                    print(f"错误读取曲线 {curve_filename}: {e}")
                    strain_stress = []

                strain_stress.append([0.0, 0.0])
                strain_stress = np.array(strain_stress, dtype=float)
                sorted_indices = np.argsort(strain_stress[:, 0])
                strain_stress_sorted = strain_stress[sorted_indices]
                _, unique_indices = np.unique(strain_stress_sorted[:, 0], return_index=True)
                strain_stress_final = strain_stress_sorted[unique_indices].tolist()

                raw_features = row[feature_cols].values.astype(float)
                standardized_features = ((raw_features - means) / stds).tolist()
                record = {
                    "Paper_Num": paper_num,
                    "Alloy_Num": alloy_num,
                    "initial_feature": standardized_features,
                    "strain_stress_curve": strain_stress_final
                }

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"处理完成！结果已保存至: {output_jsonl_path}")
        print(f"全局均值: {means.tolist()}")
        print(f"全局标准差: {stds.tolist()}")


    def standard_norm(self):
        pass

    def _load_data(self):
        csv_data = pd.read_excel(self.excell_data_path, header=1)
        csv_data = csv_data.iloc[:, 1:]
        csv_data.to_csv('./data/csv_data.csv', index=False, encoding='utf-8')

        exepriment_data = csv_data.loc[csv_data["Paper_Num"].isin(args.extract_Paper_Num)]
        exepriment_data = csv_data.loc[csv_data["Alloy_Num"].isin(args.alloy_numbers)]
        #exepriment_data.to_csv('./data/exepriment_data.csv', index=False, encoding='utf-8')

        f_list = ["Paper_Num"]+["Alloy_Num"] + args.element_list + args.processing_list
        print(f_list)
        select_data = exepriment_data.loc[:, f_list]
        data_ = self.replace_elements_with_features(select_data, args.element_list)
        return data_



    def replace_elements_with_features(self, original_df, element_list):
 
        df_comp = original_df[element_list].copy()  # shape: (n_samples, len(element_list))
        raw_feat = extract_hea_features_batch(
            df_compositions=df_comp,
            element_list=args.element_list,
            norm=False
        )
        raw_feat = np.array(raw_feat)
        assert raw_feat.shape[1] == len(args.physic_names), "特征维度与名称数量不一致"
        raw_feat = pd.DataFrame(raw_feat, index=original_df.index, columns=args.physic_names)
        print("raw_feat:", raw_feat)
        non_element_cols = [col for col in original_df.columns if col not in element_list]
        df_before = original_df[non_element_cols].copy()
        result_df = pd.concat([df_before, raw_feat], axis=1)
        return result_df



if __name__=='__main__':

    data_process = obtain_data(args)
    data_process.process_data_and_curves(
        process_csv_path="./data/origin_process_data.csv",
        curve_dir="./data/origin_curve_data",
        output_jsonl_path="./data/processed_data.jsonl"
    )

















