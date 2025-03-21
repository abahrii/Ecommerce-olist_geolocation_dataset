
"""
Notebook du modèle
Réalisé par : BAHRI Abdelghani
"""

"""
Analyse de clustering sur des données RFM
==========================================

Ce script réalise les opérations suivantes :
1. Chargement et prétraitement des données transactionnelles.
2. Calcul du résumé RFM (Récence, Fréquence, Valeur Monétaire) à l'aide de la fonction
   summary_data_from_transaction_data du package lifetimes.
3. Standardisation des variables RFM.
4. Détermination du nombre optimal de clusters via la méthode du coude.
5. Application du clustering par KMeans et visualisation en 2D et 3D.
6. Calcul du score de silhouette pour évaluer la qualité des clusters.
7. Réalisation d'un clustering hiérarchique et affichage du dendrogramme.
8. Application de DBSCAN pour détecter des clusters de forme arbitraire.
9. Calcul de l'importance relative (variation par rapport à la moyenne globale) 
   des attributs par cluster, avec visualisation sous forme de heatmap.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
from lifetimes.utils import summary_data_from_transaction_data
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D
import scipy.cluster.hierarchy as shc

# -------------------------------
# Fonctions utilitaires
# -------------------------------

def load_and_preprocess_data(file_path):
    """
    Charge les données transactionnelles et prépare un DataFrame avec l'identifiant client, la date d'achat et le prix.
    Le timestamp est converti en date (jour uniquement).
    """
    # Chargement des données (ici, le fichier est supposé être de type CSV)
    data = pd.read_csv(file_path)
    print("Aperçu des données transactionnelles :")
    print(data.head())
    
    # Extraction des colonnes utiles
    transaction_data = data[['customer_unique_id', 'order_purchase_timestamp', 'price']].copy()
    # Conversion du timestamp en date (jour)
    transaction_data['date'] = pd.to_datetime(transaction_data['order_purchase_timestamp']).dt.date
    # Suppression de la colonne initiale de timestamp
    transaction_data.drop('order_purchase_timestamp', axis=1, inplace=True)
    return transaction_data

def compute_RFM_data(transaction_data):
    """
    Calcule le résumé RFM à partir des données transactionnelles.
    Seuls les clients ayant une fréquence > 0 sont conservés.
    """
    rfm = summary_data_from_transaction_data(transaction_data, 
                                               customer_id_col='customer_unique_id', 
                                               datetime_col='date', 
                                               monetary_value_col='price')
    # Filtre sur les clients avec fréquence positive
    rfm = rfm[rfm['frequency'] > 0]
    print("Résumé RFM :")
    print(rfm.describe())
    return rfm

def plot_elbow_method(X_scaled, max_clusters=10):
    """
    Trace la courbe du WCSS (Within-Cluster Sum of Squares) pour déterminer le nombre optimal de clusters (méthode du coude).
    """
    wcss = []
    for i in range(1, max_clusters + 1):
        km = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        km.fit(X_scaled)
        wcss.append(km.inertia_)
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, max_clusters + 1), wcss, marker='8', color='red', linewidth=2)
    plt.xlabel("Nombre de clusters")
    plt.ylabel("WCSS")
    plt.title("Méthode du coude pour déterminer le nombre optimal de clusters")
    plt.grid(True)
    plt.xticks(np.arange(1, max_clusters + 1, 1))
    plt.show()

def perform_kmeans(X_scaled, n_clusters=4):
    """
    Applique l'algorithme KMeans sur les données standardisées.
    Retourne le modèle entraîné et les étiquettes de clusters.
    """
    km = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    start_time = time.time()
    labels = km.fit_predict(X_scaled)
    elapsed_time = time.time() - start_time
    print(f"KMeans (k={n_clusters}) : temps d'entraînement = {elapsed_time:.2f}s, inertie = {km.inertia_:.2f}")
    return km, labels

def plot_clusters_2D(X_scaled, labels, km_model, title="Clusters KMeans"):
    """
    Réduit la dimension des données à 2 composantes via PCA et affiche un scatter plot des clusters.
    Les centroïdes sont également affichés.
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure(figsize=(10, 6))
    unique_labels = np.unique(labels)
    colors = sns.color_palette("bright", len(unique_labels))
    for i, label in enumerate(unique_labels):
        plt.scatter(X_pca[labels == label, 0], X_pca[labels == label, 1], s=50, color=colors[i], label=f'Cluster {label+1}')
    centers_pca = pca.transform(km_model.cluster_centers_)
    plt.scatter(centers_pca[:, 0], centers_pca[:, 1], s=200, c='black', marker='s', label='Centroids')
    plt.title(title)
    plt.xlabel("Composante principale 1")
    plt.ylabel("Composante principale 2")
    plt.legend()
    plt.show()

def plot_clusters_3D(X_scaled, labels, km_model, title="Clusters KMeans 3D"):
    """
    Réduit la dimension des données à 3 composantes via PCA et affiche un scatter plot 3D des clusters.
    Les centroïdes sont affichés dans l'espace réduit.
    """
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    unique_labels = np.unique(labels)
    colors = sns.color_palette("bright", len(unique_labels))
    for i, label in enumerate(unique_labels):
        ax.scatter(X_pca[labels == label, 0], X_pca[labels == label, 1], X_pca[labels == label, 2], 
                   s=50, color=colors[i], label=f'Cluster {label+1}')
    centers_pca = pca.transform(km_model.cluster_centers_)
    ax.scatter(centers_pca[:, 0], centers_pca[:, 1], centers_pca[:, 2], s=200, c='black', marker='s', label='Centroids')
    ax.set_title(title)
    ax.set_xlabel("Composante 1")
    ax.set_ylabel("Composante 2")
    ax.set_zlabel("Composante 3")
    plt.legend()
    plt.show()

def compute_silhouette_scores(X_scaled, cluster_range=range(2, 10)):
    """
    Calcule et trace le score de silhouette pour différents nombres de clusters.
    Le score de silhouette mesure la qualité du clustering.
    """
    scores = []
    for k in cluster_range:
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=0)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores.append(score)
    plt.figure(figsize=(8, 4))
    plt.plot(list(cluster_range), scores, marker='o')
    plt.xlabel("Nombre de clusters")
    plt.ylabel("Silhouette Score")
    plt.title("Évaluation du score de silhouette pour KMeans")
    plt.grid(True)
    plt.show()

def perform_hierarchical_clustering(X_scaled, n_clusters=4):
    """
    Effectue un clustering hiérarchique.
    Affiche d'abord le dendrogramme, puis retourne les étiquettes de clusters obtenues avec l'algorithme Agglomeratif.
    """
    plt.figure(figsize=(10, 7))
    plt.title("Dendrogramme pour clustering hiérarchique")
    shc.dendrogram(shc.linkage(X_scaled, method='ward'))
    plt.xlabel("Observations")
    plt.ylabel("Distance")
    plt.show()
    
    cluster_model = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
    labels = cluster_model.fit_predict(X_scaled)
    return labels

def perform_dbscan(X_scaled, eps=0.7, min_samples=8):
    """
    Applique l'algorithme DBSCAN pour détecter des clusters de forme arbitraire.
    Retourne les étiquettes de clusters.
    """
    dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)
    start_time = time.time()
    labels = dbscan_model.fit_predict(X_scaled)
    elapsed_time = time.time() - start_time
    print(f"DBSCAN : temps d'entraînement = {elapsed_time:.2f}s")
    return labels

def compute_relative_importance(rfm, labels):
    """
    Calcule l'importance relative des attributs par cluster.
    Pour chaque attribut, compare la moyenne du cluster à la moyenne globale.
    Affiche ensuite les résultats sous forme de heatmap.
    """
    rfm_with_cluster = rfm.copy()
    rfm_with_cluster["Cluster"] = labels
    cluster_avg = rfm_with_cluster.groupby("Cluster").mean()
    overall_avg = rfm_with_cluster.mean()
    relative_imp = (cluster_avg / overall_avg) - 1
    print("Importance relative (variation par rapport à la moyenne globale) :")
    print(relative_imp.round(2))
    plt.figure(figsize=(8, 2))
    sns.heatmap(relative_imp, annot=True, fmt=".2f", cmap="RdYlGn")
    plt.title("Importance relative des attributs par cluster")
    plt.show()

# -------------------------------
# Fonction principale
# -------------------------------

if __name__ == "__main__":
    # 1. Chargement et prétraitement des données transactionnelles
    file_path = "cleaned_data_tr.csv" # Chemin vers le fichier CSV
    transaction_data = load_and_preprocess_data(file_path)
    
    # 2. Calcul du résumé RFM
    rfm_data = compute_RFM_data(transaction_data)
    
    # 3. Sélection des variables RFM pour le clustering
    # On conserve les colonnes 'recency', 'frequency' et 'monetary_value'
    data_RFM = rfm_data[['recency', 'frequency', 'monetary_value']]
    print("Aperçu des données RFM :")
    print(data_RFM.head())
    
    # 4. Standardisation des données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data_RFM)
    
    # 5. Méthode du coude pour déterminer le nombre optimal de clusters
    plot_elbow_method(X_scaled, max_clusters=10)
    
    # 6. Application de KMeans (par exemple, avec 4 clusters)
    km_model, kmeans_labels = perform_kmeans(X_scaled, n_clusters=4)
    
    # 7. Visualisation des clusters en 2D et en 3D (après réduction par PCA)
    plot_clusters_2D(X_scaled, kmeans_labels, km_model, title="Clusters KMeans (RFM) en 2D")
    plot_clusters_3D(X_scaled, kmeans_labels, km_model, title="Clusters KMeans (RFM) en 3D")
    
    # 8. Évaluation du clustering via le score de silhouette
    compute_silhouette_scores(X_scaled, cluster_range=range(2, 10))
    
    # 9. Clustering hiérarchique
    hierarchical_labels = perform_hierarchical_clustering(X_scaled, n_clusters=4)
    plot_clusters_2D(X_scaled, hierarchical_labels, km_model, title="Clusters Hiérarchiques (via PCA)")
    
    # 10. Clustering avec DBSCAN
    dbscan_labels = perform_dbscan(X_scaled, eps=0.7, min_samples=8)
    plot_clusters_2D(X_scaled, dbscan_labels, km_model, title="Clusters DBSCAN (via PCA)")
    
    # 11. Calcul de l'importance relative des clusters pour les données RFM
    compute_relative_importance(data_RFM, kmeans_labels)


