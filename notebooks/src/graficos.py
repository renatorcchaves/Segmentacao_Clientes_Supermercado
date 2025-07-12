import os
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from matplotlib.colors import ListedColormap

def grafico_elbow_silhouette(X, random_state=42, intervalo_k=(2, 11)):
    
    # OBS: O "X" precisa ser um dataframe só com valores numéricos, colunas categóricas precisam ter passado por preprocessamento.

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5), tight_layout=True)

    elbow = {}
    silhouette = []

    k_range = range(*intervalo_k)

    for numero in k_range:
        kmeans = KMeans(n_clusters=numero, n_init= 10, random_state=random_state)
        kmeans.fit(X)         

        elbow[numero] = kmeans.inertia_      # inertia_ : soma da distância quadrada de cada ponto para o centroide de seu cluster
        labels = kmeans.labels_              # labels_: nome de cada cluster
        
        silhouette.append(silhouette_score(X, labels))       
        # silhouete_score: dentro de cada cluster ele calcula a distância média de cada ponto, e compara com a distânia do ponto pro cluster mais próximo que ele não pertence
        # silhouete_score: varia de -1 a 1, quanto maior melhor, mais bem dividido estão os clusters. Valor igual a 0 quer dizer que o ponto está na distancia igual entre o centro de 2 cluster
        # silhouette_score(kmeans.transform(X)): pra calcular esse score o X precisa estar transformado pelas etapas de preprocessamento do kmeans 

    # OBS: lineplot precisa dos valores em formato de lista
    sns.lineplot( x = list(elbow.keys()), y = list(elbow.values()), ax=ax[0], marker='o')
    ax[0].set_title('Elbow Method')
    ax[0].set_xlabel('K')
    ax[0].set_ylabel('Inertia')

    sns.lineplot( x = list(k_range), y = silhouette, ax=ax[1], marker='o')
    ax[1].set_title('Silhouette Method')
    ax[1].set_xlabel('K')
    ax[1].set_ylabel('Silhouette Score')

    fig.suptitle('Definição do número de Clusters', fontsize=15, fontweight='bold')

    plt.show()


def visualizar_clusters_3d(
    dataframe,                       # Precisamos passar o dataframe preprocessado, pois os centroides foram calculados em cima desse dataframe preprocessado
    colunas,                         # Informar em formato de lista, e o nome delas precisa ter o prefixo do preprocessamento 'one_hot_coluna' ou 'standard_coluna'
    quantidade_cores_clusters,       # Vai ser sempre igual a quantidade de clusters (e não das colunas)
    centroids,                       # Esse parametro deve ser fornecido, e pode ter apenas os centroides das colunas que serão exibidas no gráfico
    mostrar_centroids=True, 
    mostrar_pontos=False,            # Se for usado o parametro igual a True, precisa ser fornecido a 'coluna_cluster' abaixo
    coluna_clusters=None,            # Esse dataframe['coluna_cluster'] não precisa ser o mesmo df_preprocessado do 1º parametro, pode ser o df_clustered['cluster']
):

    fig = plt.figure()
    
    ax = fig.add_subplot(111, projection="3d")
    
    cores = plt.cm.tab10.colors[:quantidade_cores_clusters]
    cores = ListedColormap(cores)
    
    x = dataframe[colunas[0]]
    y = dataframe[colunas[1]]
    z = dataframe[colunas[2]]
    
    ligar_centroids = mostrar_centroids
    ligar_pontos = mostrar_pontos
    
    for i, centroid in enumerate(centroids):
        if ligar_centroids: 
            ax.scatter(*centroid, s=500, alpha=0.5)
            ax.text(*centroid, f"{i}", fontsize=20, horizontalalignment="center", verticalalignment="center")
    
        if ligar_pontos:
            s = ax.scatter(x, y, z, c=coluna_clusters, cmap=cores)
            ax.legend(*s.legend_elements(), bbox_to_anchor=(1.3, 1))
    
    ax.set_xlabel(colunas[0])
    ax.set_ylabel(colunas[1])
    ax.set_zlabel(colunas[2])
    ax.set_title("Clusters")
    
    plt.show()