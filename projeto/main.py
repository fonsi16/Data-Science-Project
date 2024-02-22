# %% Graficos para visualizar dados
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def disease_countplots(data):
    # Criar um conjunto de subplots
    fig, axes = plt.subplots(nrows=data.shape[1], ncols=1, figsize=(20, 13 * data.shape[1]))
    for i in range(len(data.columns)):
        col = data.columns[i]
        sns.countplot(x=col, data=data, hue="HeartDisease", palette="seismic", ax=axes[i])
    axes[0].legend([])
    plt.show()


df = pd.read_csv('data/heart_2020_cleaned.csv')

df.info()

disease_countplots(df)
