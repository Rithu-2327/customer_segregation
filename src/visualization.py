import matplotlib.pyplot as plt
import seaborn as sns

def plot_clusters(df, x_col, y_col, hue_col='Cluster Name', save_path=None):
    """Scatter plot of clusters"""
    plt.figure(figsize=(8,6))
    sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=hue_col,
        palette='Set1',
        s=60
    )
    plt.title(f'{y_col} vs {x_col} Clusters')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend(title='Cluster')
    if save_path:
        plt.savefig(save_path)
    plt.show()