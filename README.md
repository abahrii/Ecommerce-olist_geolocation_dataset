# Analyse de Clustering eCommerce

Ce repository contient plusieurs programmes permettant d'analyser des données transactionnelles e-commerce en utilisant des techniques de clustering basées sur le modèle RFM (Récence, Fréquence, Valeur Monétaire).

## Contenu du Repository

- **Notebook_analyse_ecomerce.ipynb**  
  Notebook Jupyter détaillant l'analyse de données e-commerce à travers ces techniques de clustering. 
  
- **Clustering_ecommerce.py**  
  Ce script réalise l'analyse de clustering sur des données transactionnelles en suivant les étapes suivantes :
  - Chargement et prétraitement des données (CSV).
  - Calcul du résumé RFM à l'aide de la fonction `summary_data_from_transaction_data` du package [lifetimes](https://lifetimes.readthedocs.io/en/latest/) :contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}.
  - Standardisation des variables RFM.
  - Détermination du nombre optimal de clusters via la méthode du coude.
  - Application de l'algorithme KMeans avec visualisations en 2D et 3D (réduction par PCA).
  - Évaluation de la qualité des clusters grâce au score de silhouette.
  - Réalisation d'un clustering hiérarchique avec affichage du dendrogramme.
  - Utilisation de l'algorithme DBSCAN pour détecter des clusters de formes variées.
  - Calcul de l'importance relative des attributs par cluster et affichage sous forme de heatmap.

- **interface_ecom.py**  
  Application interactive développée avec [Streamlit](https://streamlit.io/) qui permet de :
  - Charger un fichier CSV contenant les données transactionnelles.
  - Prétraiter et calculer le résumé RFM.
  - Choisir entre plusieurs méthodes de clustering (KMeans, clustering hiérarchique, DBSCAN) via une interface conviviale.
  - Visualiser les clusters en 2D après réduction de dimension avec PCA.
  - Afficher le score de silhouette pour évaluer la qualité du clustering.


## Prérequis

Pour exécuter ces programmes, assurez-vous d'avoir installé Python 3.x ainsi que les bibliothèques suivantes :

- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [scipy](https://www.scipy.org/)
- [lifetimes](https://lifetimes.readthedocs.io/en/latest/)
- [streamlit](https://streamlit.io/)

Installez-les via pip : ```bash

pip install pandas numpy matplotlib seaborn scikit-learn scipy lifetimes streamlit

Exécution des Programmes
1. Script de Clustering
Exécutez le script en ligne de commande :

bash
Copier
python Clustering_ecommerce.py
Assurez-vous que le fichier CSV (cleaned_data_tr.csv) contenant les données transactionnelles se trouve dans le même répertoire.

2. Application Streamlit
Lancez l'application avec la commande suivante :

bash
Copier
streamlit run interface_ecom.py
Vous pourrez alors charger un fichier CSV et interagir avec les différentes méthodes de clustering via l'interface.

3. Notebook d'Analyse
Le notebook Notebook_analyse_ecomerce.ipynb propose une analyse détaillée et interactive des données e-commerce. Ouvrez-le dans Jupyter Notebook ou JupyterLab pour explorer et reproduire l'analyse.

Visualisations et Résultats
Les images ci-dessous illustrent quelques résultats obtenus avec ces programmes :

Olist PowerBI Campagne Communication

Visualisation des Clusters KMeans
