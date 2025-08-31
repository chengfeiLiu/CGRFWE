import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from sklearn.feature_selection import mutual_info_regression
from scipy.signal import coherence
from scipy.fft import fft
from itertools import combinations

df = pd.read_excel('1719751999.xlsx')
df_time = df.iloc[:, 0]
df_data = df.iloc[:, 1:9]

pearson_corr = df_data.corr(method='pearson')
spearman_corr = df_data.corr(method='spearman')

def cross_corr_lag1(x, y):
    return np.corrcoef(x[:-1], y[1:])[0, 1]

lag_corr_matrix = pd.DataFrame(index=df_data.columns, columns=df_data.columns)
for i, j in combinations(df_data.columns, 2):
    val = cross_corr_lag1(df_data[i].values, df_data[j].values)
    lag_corr_matrix.loc[i, j] = val
    lag_corr_matrix.loc[j, i] = val
np.fill_diagonal(lag_corr_matrix.values, 1)

def compute_coherence_matrix(df_data):
    coherence_matrix = pd.DataFrame(index=df_data.columns, columns=df_data.columns)
    for i, j in combinations(df_data.columns, 2):
        f, Cxy = coherence(df_data[i], df_data[j])
        coherence_matrix.loc[i, j] = coherence_matrix.loc[j, i] = np.mean(Cxy)
    np.fill_diagonal(coherence_matrix.values, 1)
    return coherence_matrix.astype(float)

coherence_corr = compute_coherence_matrix(df_data)

mi_matrix = pd.DataFrame(index=df_data.columns, columns=df_data.columns)
for i in df_data.columns:
    mi = mutual_info_regression(df_data, df_data[i])
    mi_matrix[i] = mi
mi_matrix = mi_matrix.astype(float)
mi_matrix.index = df_data.columns

with pd.ExcelWriter('correlation_results.xlsx') as writer:
    pearson_corr.to_excel(writer, sheet_name='Pearson')
    spearman_corr.to_excel(writer, sheet_name='Spearman')
    lag_corr_matrix.to_excel(writer, sheet_name='Lag1_CrossCorr')
    coherence_corr.to_excel(writer, sheet_name='Coherence')
    mi_matrix.to_excel(writer, sheet_name='Mutual_Information')

def plot_heatmap(matrix, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix.astype(float), annot=True, cmap='coolwarm', fmt=".2f", square=True)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

plot_heatmap(pearson_corr, "Pearson Correlation")
plot_heatmap(spearman_corr, "Spearman Correlation")
plot_heatmap(lag_corr_matrix, "Lag-1 Cross Correlation")
plot_heatmap(coherence_corr, "Coherence (Frequency Domain)")
plot_heatmap(mi_matrix, "Mutual Information")
