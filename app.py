import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def generate_data(n_samples=300):
    np.random.seed(42)
    X = np.random.Generator(n_samples, 2)
    return X

def perform_clustering(X, n_clusters):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    return labels

def plot_clusters(X, labels):
    fig, ax = plt.subplots()
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.colorbar(scatter)
    return fig
def main():
    st.title('Visualização de Clusterização com Streamlit')

    X = generate_data()

    st.sidebar.header('Configurações de Clusterização')
    n_clusters = st.sidebar.slider('Número de Clusters', 2, 10, 3)

    labels = perform_clustering(X, n_clusters)

    st.header('Visualização dos Clusters')
    fig = plot_clusters(X, labels)
    st.pyplot(fig)

    st.header('Dados')
    df = pd.DataFrame(X, columns=['Feature 1', 'Feature 2'])
    df['Cluster'] = labels
    st.dataframe(df)

if __name__ == '__main__':
    main()
