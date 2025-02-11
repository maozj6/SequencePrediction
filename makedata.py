import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def process_csv_folder(input_folder, output_folder, step=50):
    """
    读取文件夹中的所有CSV文件，每隔50行采样数据，
    将3-5列放入一个list，6-26列放入另外一个list，并保存为 .npz 文件。

    Args:
        input_folder (str): CSV 文件夹路径
        output_folder (str): 处理后 .npz 文件的保存路径
        step (int): 采样间隔（每 step 行取一行）
    """
    os.makedirs(output_folder, exist_ok=True)

    # 遍历所有 CSV 文件
    for file_name in tqdm(os.listdir(input_folder)):
        if file_name.endswith(".csv"):
            file_path = os.path.join(input_folder, file_name)
            df = pd.read_csv(file_path)

            # 确保列数足够
            if df.shape[1] < 27:
                print(f"Skipping {file_name}: Not enough columns.")
                continue

            # 直接每隔 50 行取一条数据
            sampled_df = df.iloc[::step, :]

            # 提取目标列
            group1 = sampled_df.iloc[:, 3:6].to_numpy()  # 3-5列
            group2 = sampled_df.iloc[:, 6:20].to_numpy()  # 6-26列

            # 保存到 .npz 文件
            output_file = os.path.join(output_folder, file_name.replace(".csv", ".npz"))
            np.savez(output_file, state=group1, obs=group2)

            print(f"Processed and saved: {output_file}")


if __name__ == '__main__':

    # 使用示例
    input_folder = "./new_preprocessed_data/new_preprocessed_data/"  # CSV 文件所在的文件夹
    output_folder = "./newdata/"  # 输出 npz 文件的文件夹
    process_csv_folder(input_folder, output_folder)
