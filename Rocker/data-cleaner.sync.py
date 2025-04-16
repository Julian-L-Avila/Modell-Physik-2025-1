# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv('./data/003-data.tsv', sep='\t')
df2 = pd.read_csv('./data/test-data.tsv', sep='\t')

df_merged = pd.concat([df1, df2], axis=1)

df = df_merged

df.iloc[:, 3] *= -1
df.iloc[:, 3] += 1.5

df.to_csv('./data/merged.tsv', sep='\t', index=False)

x1, y1 = df.iloc[:, 0], df.iloc[:, 1]
x2, y2 = df.iloc[:, 2], df.iloc[:, 3]

plt.plot(x1, y1, label='file1: col1 vs col2')
plt.plot(x2, y2, label='file2: col1 vs col2')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.tight_layout()
plt.show()
