import os
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
import torch

class ExcelSeqLoader(Dataset):
    def __init__(self, folder_path, data_columns, label_column, seq_len=100, step=1, flag='train',scaler=None):
        self.folder_path = folder_path
        self.data_columns = data_columns
        self.label_column = label_column
        self.seq_len = seq_len
        self.step = step
        self.flag = flag
        self.scaler = scaler

        # Load and preprocess all sequences
        sequences, labels = self.load_all_data()
        self.labels = np.array(labels)

        # Train/val/test split (80/10/10)
        total_len = len(sequences)
        train_end = int(0.8 * total_len)
        val_end = int(0.9 * total_len)

        if flag == 'train':
            self.data = sequences[:train_end]
            self.labels = labels[:train_end]
        elif flag == 'val':
            self.data = sequences[train_end:val_end]
            self.labels = labels[train_end:val_end]
        elif flag == 'test':
            self.data = sequences[val_end:]
            self.labels = labels[val_end:]
        else:
            raise ValueError(f"Invalid flag: {flag}")

        # Fit scaler only on training set
        if flag == 'train':
            print("Before scaling:", np.array(self.data).shape)
            print("Feature seq shape[0] =", self.data[0].shape[0])  # C
            print("Expected channels =", len(self.data_columns))
            all_data = np.concatenate(self.data, axis=1).T  # shape: (N*T, C)
            self.scaler.fit(all_data)

        # Normalize all sequences
        self.data = [self.scaler.transform(seq.T).T for seq in self.data]  # keep (C, T)

        print(f"[{flag}] Total samples: {len(self.data)}")

    def load_all_data(self):
        raw_sequences = []
        raw_labels = []

        for filename in os.listdir(self.folder_path):
            if filename.endswith(".xlsx") or filename.endswith(".xls"):
                filepath = os.path.join(self.folder_path, filename)
                df = pd.read_excel(filepath)

                if not all(col in df.columns for col in self.data_columns + [self.label_column]):
                    print(f"❌ 文件 {filename} 缺少必要列，跳过。")
                    continue

                df = df.sort_values(by='time')
                features = df[self.data_columns].copy()
                labels = df[self.label_column].copy()

                for i in range(0, len(df) - self.seq_len + 1, self.step):
                    feature_seq = features.iloc[i:i+self.seq_len].values.T  # shape: (C, T)
                    label_seq = labels.iloc[i:i+self.seq_len].values
                    label_majority = int(np.round(np.mean(label_seq)))  # 多数标签

                    raw_sequences.append(feature_seq)
                    raw_labels.append(label_majority)

        return raw_sequences, raw_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        feature = np.float32(self.data[index]).T
        label = np.int64(self.labels[index])
        return feature, label
