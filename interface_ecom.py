import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from lifetimes.utils import summary_data_from_transaction_data
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as shc

# Configuration de Streamlit
st.title("Interface d'Analyse de Clustering RFM")
st.write("Cette application vous permet d'analyser des données RFM et d'appliquer différents algorithmes de clustering.")

# Téléchargement du fichier CSV
uploaded_file = st.file_uploader("Charger le fichier CSV des données transactionnelles", type=["csv"])
if uploaded_file is not None:
    try:
        # Lecture du fichier téléchargé
        transaction_data = pd.read_csv(uploaded_file, encoding="utf-8")
    except Exception as e:
        st.error("Erreur lors du chargement du fichier : " + str(e))
    
    # Vérification des colonnes requises
    required_columns = ['customer_unique_id', 'order_purchase_timestamp', 'price']
    if all(col in transaction_data.columns for col in required_columns):
        st.write("Aperçu des données :", transaction_data.head())
        # Prétraitement des données
        transaction_data['date'] = pd.to_datetime(transaction_data['order_purchase_timestamp'], errors='coerce').dt.date
        transaction_data.drop('order_purchase_timestamp', axis=1, inplace=True)
        st.write("Données prétraitées :", transaction_data.head())
        
        # Calcul du résumé RFM
        rfm = summary_data_from_transaction_data(transaction_data, 
                                                 customer_id_col='customer_unique_id', 
                                                 datetime_col='date', 
                                                 monetary_value_col='price')
        rfm = rfm[rfm['frequency'] > 0]
        st.write("Résumé RFM :", rfm.describe())
        
        # Sélection des variables RFM
        data_RFM = rfm[['recency', 'frequency', 'monetary_value']]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data_RFM)
        
        # Choix de la méthode de clustering
        method = st.sidebar.selectbox("Méthode de clustering", ["KMeans", "Clustering Hiérarchique", "DBSCAN"])
        
        if method == "KMeans":
            n_clusters = st.sidebar.number_input("Nombre de clusters", min_value=2, max_value=10, value=4)
            km = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
            labels = km.fit_predict(X_scaled)
            st.write(f"KMeans appliqué avec {n_clusters} clusters.")
            
            # Visualisation 2D avec PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            fig, ax = plt.subplots(figsize=(8,6))
            for i in range(n_clusters):
                ax.scatter(X_pca[labels == i, 0], X_pca[labels == i, 1], s=50, label=f'Cluster {i+1}')
            centers_pca = pca.transform(km.cluster_centers_)
            ax.scatter(centers_pca[:, 0], centers_pca[:, 1], s=200, c='black', marker='s', label='Centroids')
            ax.set_xlabel("Composante 1")
            ax.set_ylabel("Composante 2")
            ax.set_title("Clusters KMeans en 2D")
            ax.legend()
            st.pyplot(fig)
            
            score = silhouette_score(X_scaled, labels)
            st.write(f"Silhouette Score : {score:.2f}")
        
        elif method == "Clustering Hiérarchique":
            n_clusters = st.sidebar.number_input("Nombre de clusters (Hiérarchique)", min_value=2, max_value=10, value=4)
            # Affichage du dendrogramme
            fig, ax = plt.subplots(figsize=(10,7))
            shc.dendrogram(shc.linkage(X_scaled, method='ward'), ax=ax)
            ax.set_title("Dendrogramme")
            ax.set_xlabel("Observations")
            ax.set_ylabel("Distance")
            st.pyplot(fig)
            
            agg = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
            labels = agg.fit_predict(X_scaled)
            st.write(f"Clustering hiérarchique appliqué avec {n_clusters} clusters.")
            
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            fig, ax = plt.subplots(figsize=(8,6))
            for i in np.unique(labels):
                ax.scatter(X_pca[labels == i, 0], X_pca[labels == i, 1], s=50, label=f'Cluster {i+1}')
            ax.set_title("Clustering Hiérarchique en 2D")
            ax.set_xlabel("Composante 1")
            ax.set_ylabel("Composante 2")
            ax.legend()
            st.pyplot(fig)
            
            score = silhouette_score(X_scaled, labels)
            st.write(f"Silhouette Score : {score:.2f}")
        
        elif method == "DBSCAN":
            eps = st.sidebar.number_input("Paramètre eps", min_value=0.1, max_value=10.0, value=0.7, step=0.1)
            min_samples = st.sidebar.number_input("Min Samples", min_value=1, max_value=20, value=8)
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X_scaled)
            st.write("Clustering DBSCAN appliqué.")
            
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            fig, ax = plt.subplots(figsize=(8,6))
            for i in np.unique(labels):
                ax.scatter(X_pca[labels == i, 0], X_pca[labels == i, 1], s=50, label=f'Cluster {i}')
            ax.set_title("Clusters DBSCAN en 2D")
            ax.set_xlabel("Composante 1")
            ax.set_ylabel("Composante 2")
            ax.legend()
            st.pyplot(fig)
            try:
                score = silhouette_score(X_scaled, labels)
                st.write(f"Silhouette Score : {score:.2f}")
            except Exception as e:
                st.write("Silhouette Score non calculable :", e)
    else:
        st.error("Le fichier doit contenir les colonnes 'customer_unique_id', 'order_purchase_timestamp' et 'price'.")
