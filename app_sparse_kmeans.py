from sklearn.base import BaseEstimator, ClusterMixin
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Configuração da página
st.set_page_config(page_title="Visualização de Acidentes Ferroviários - ANTT")


# Funções de carregamento e pré-processamento de dados
@st.cache_data
def load_data():
    df = pd.read_csv(
        "datasets/acidents_ferroviarios_2004_2024_com_coords.csv",
        encoding="UTF-8",
        sep=";",
    )
    df["Data_Ocorrencia"] = pd.to_datetime(df["Data_Ocorrencia"], format="mixed")
    df["Quilometro_Inicial"] = pd.to_numeric(
        df["Quilometro_Inicial"].replace(",", "."), errors="coerce"
    )
    df["Hora_Ocorrencia"] = pd.to_datetime(
        df["Hora_Ocorrencia"], format="%H:%M"
    ).dt.time
    df["Mercadoria"] = df["Mercadoria"].fillna("Não Identificada")
    df["Prejuizo_Financeiro"] = pd.to_numeric(df['Prejuizo_Financeiro'].str.replace(',','.'), errors='coerce')
    df["Prejuizo_Financeiro"] = df["Prejuizo_Financeiro"].fillna(0)

    return df


def preprocess_for_kmeans(df):
    city_data = (
        df.groupby("Municipio")
        .agg(
            {
                "Latitude": "first",
                "Longitude": "first",
                "Data_Ocorrencia": "count",
            }
        )
        .reset_index()
    )
    city_data.rename(columns={"Data_Ocorrencia": "num_acidentes"}, inplace=True)
    return city_data


def preprocess_for_sparse_kmeans(df):
    # Filtrar dados de dezembro de 2020 a julho de 2024
    df = df[
        (df["Data_Ocorrencia"] >= "2020-12-01")
        & (df["Data_Ocorrencia"] <= "2024-07-31")
    ]

    city_data = (
        df.groupby("Municipio")
        .agg(
            {
                "Latitude": "first",
                "Longitude": "first",
                "Data_Ocorrencia": "count",
                "Interrupcao": "sum",
                "Prejuizo_Financeiro": "sum",
                "Mercadoria": "first",
            }
        )
        .reset_index()
    )
    city_data.rename(columns={"Data_Ocorrencia": "num_acidentes"}, inplace=True)
    return city_data


class SparseKMeans(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=5, random_state=None, lasso_weight=0.1):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.lasso_weight = lasso_weight

    def fit_predict(self, X):
        # Normalize features
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        # Apply L1 regularization (Lasso) to feature weights
        feature_weights = np.ones(X_scaled.shape[1])
        for i in range(X_scaled.shape[1]):
            variance = np.var(X_scaled[:, i])
            feature_weights[i] = max(0, variance - self.lasso_weight)

        # Apply feature weights to data
        X_weighted = X_scaled * feature_weights

        # Perform regular k-means on weighted data
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.labels_ = kmeans.fit_predict(X_weighted)

        # Store cluster centers
        self.cluster_centers_ = kmeans.cluster_centers_
        self.feature_weights_ = feature_weights

        return self.labels_

    def fit(self, X):
        self.fit_predict(X)
        return self


def create_cluster_map(city_data, n_clusters):
    scaler = StandardScaler()
    features = city_data[["num_acidentes", "Latitude", "Longitude"]]
    normalized_features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    city_data["cluster"] = kmeans.fit_predict(normalized_features)

    # Transformar os centroides normalizados de volta para a escala original
    centroids_normalized = kmeans.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_normalized)

    # Criar DataFrame com os centroides
    centroids_df = pd.DataFrame(
        centroids_original, columns=["num_acidentes", "Latitude", "Longitude"]
    )

    # Calcular estatísticas por cluster
    cluster_stats = (
        city_data.groupby("cluster")
        .agg(
            {
                "num_acidentes": ["mean", "count"],
                "Municipio": lambda x: ", ".join(x.head(3)),  # Top 3 cidades do cluster
            }
        )
        .reset_index()
    )

    cluster_stats.columns = [
        "cluster",
        "acidentes_medios",
        "num_cidades",
        "principais_cidades",
    ]
    centroids_df["cluster"] = cluster_stats["cluster"]
    centroids_df["acidentes_medios"] = cluster_stats["acidentes_medios"]
    centroids_df["num_cidades"] = cluster_stats["num_cidades"]
    centroids_df["principais_cidades"] = cluster_stats["principais_cidades"]

    fig = px.scatter_mapbox(
        centroids_df,
        lat="Latitude",
        lon="Longitude",
        size="acidentes_medios",
        color="acidentes_medios",
        hover_data=["acidentes_medios", "num_cidades", "principais_cidades"],
        zoom=4,
        height=900,
        title="Centroides dos Clusters (K-Means)",
        color_continuous_scale=px.colors.sequential.Oranges[
            3:
        ],
    )

    fig.update_layout(
        mapbox_style="carto-darkmatter",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    fig.update_traces(marker=dict(size=20))

    return fig


def create_sparse_kmeans_map(city_data, n_clusters):
    scaler = StandardScaler()
    features_columns = ["num_acidentes", "Latitude", "Longitude", "Interrupcao", "Prejuizo_Financeiro"]
    features = city_data[features_columns]
    normalized_features = scaler.fit_transform(features)

    sparse_kmeans = SparseKMeans(n_clusters=n_clusters, random_state=42)
    city_data["cluster"] = sparse_kmeans.fit_predict(normalized_features)

    # Transformar os centroides normalizados de volta para a escala original
    centroids_normalized = sparse_kmeans.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_normalized)

    # Criar DataFrame com os centroides
    centroids_df = pd.DataFrame(
        centroids_original, columns=features_columns
    )

    # Calcular estatísticas por cluster
    cluster_stats = (
        city_data.groupby("cluster")
        .agg(
            {
                "num_acidentes": ["mean", "count"],
                "Municipio": lambda x: ", ".join(x.head(3)),  # Top 3 cidades do cluster
            }
        )
        .reset_index()
    )

    cluster_stats.columns = [
        "cluster",
        "acidentes_medios",
        "num_cidades",
        "principais_cidades",
    ]
    centroids_df["cluster"] = cluster_stats["cluster"]
    centroids_df["acidentes_medios"] = cluster_stats["acidentes_medios"]
    centroids_df["num_cidades"] = cluster_stats["num_cidades"]
    centroids_df["principais_cidades"] = cluster_stats["principais_cidades"]

    # Adicionar informação sobre os pesos das features
    feature_importance = pd.DataFrame(
        {
            "feature": features_columns,
            "weight": sparse_kmeans.feature_weights_,
        }
    )

    fig = px.scatter_mapbox(
        centroids_df,
        lat="Latitude",
        lon="Longitude",
        size="acidentes_medios",
        color="acidentes_medios",
        hover_data=["acidentes_medios", "num_cidades", "principais_cidades"],
        zoom=4,
        height=900,
        title="Centroides dos Clusters (Sparse K-Means)",
        color_continuous_scale=px.colors.sequential.Oranges[
            3:
        ],
    )

    fig.update_layout(
        mapbox_style="carto-darkmatter",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    fig.update_traces(marker=dict(size=20))

    return fig, feature_importance


def create_accident_map(df):
    df_map = df.dropna(subset=["Municipio", "Latitude", "Longitude"])
    if not df_map.empty:
        fig_map = px.scatter_mapbox(
            df_map,
            lat="Latitude",
            lon="Longitude",
            hover_name="Municipio",
            hover_data=[
                "Latitude",
                "Longitude",
                "Concessionaria",
                "Data_Ocorrencia",
                "Linha",
                "Mercadoria",
            ],
            zoom=3,
            height=600,
            width=600,
            color_discrete_sequence=["red"],
        )
        fig_map.update_layout(mapbox_style="open-street-map")
        fig_map.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        return fig_map
    return None


def create_bar_chart(df, x, y, title):
    fig = px.bar(df, x=x, y=y, labels={x: x, y: "Número de Acidentes"}, title=title)
    return fig


def create_time_series(df):
    df_time = df.groupby("Data_Ocorrencia").size().reset_index(name="Numero_Acidentes")
    fig = px.line(
        df_time,
        x="Data_Ocorrencia",
        y="Numero_Acidentes",
        labels={"Data_Ocorrencia": "Data", "Numero_Acidentes": "Número de Acidentes"},
    )
    return fig


def create_pie_chart(df, column, title):
    df_pie = df[column].value_counts().reset_index()
    df_pie.columns = [column, "Numero_Acidentes"]
    fig = px.pie(
        df_pie,
        values="Numero_Acidentes",
        names=column,
        title=title,
    )
    return fig


# Função principal
def main():
    st.title("Visualização de Acidentes Ferroviários - ANTT")

    # Carregamento de dados
    df = load_data()
    # Sidebar para filtros
    st.sidebar.header("Filtros")
    concessionaria = st.sidebar.multiselect(
        "Concessionária", options=df["Concessionaria"].unique()
    )
    uf = st.sidebar.multiselect("UF", options=df["UF"].unique())
    mercadoria = st.sidebar.multiselect("Mercadoria", options=df["Mercadoria"].unique())
    date_range = st.sidebar.date_input(
        "Intervalo de Data",
        [df["Data_Ocorrencia"].min().date(), df["Data_Ocorrencia"].max().date()],
        min_value=df["Data_Ocorrencia"].min().date(),
        max_value=df["Data_Ocorrencia"].max().date(),
    )
    time_range = st.sidebar.slider(
        "Intervalo de Hora",
        value=(df["Hora_Ocorrencia"].min(), df["Hora_Ocorrencia"].max()),
        format="HH:mm",
    )
    n_clusters = st.sidebar.slider(
        "Escolha o número de clusters", min_value=2, max_value=100, value=5
    )

    # Aplicação dos filtros
    df_filtered = df.copy()
    if concessionaria:
        df_filtered = df_filtered[df_filtered["Concessionaria"].isin(concessionaria)]
    if uf:
        df_filtered = df_filtered[df_filtered["UF"].isin(uf)]
    if mercadoria:
        df_filtered = df_filtered[df_filtered["Mercadoria"].isin(mercadoria)]
    df_filtered = df_filtered[
        (df_filtered["Data_Ocorrencia"].dt.date >= date_range[0])
        & (df_filtered["Data_Ocorrencia"].dt.date <= date_range[1])
    ]
    df_filtered = df_filtered[
        (df_filtered["Hora_Ocorrencia"] >= time_range[0])
        & (df_filtered["Hora_Ocorrencia"] <= time_range[1])
    ]

    # Visualizações de cluster e mapa
    st.header("Clusterização de Acidentes por Cidade (K-Means)")
    city_data = preprocess_for_kmeans(df_filtered)
    st.plotly_chart(create_cluster_map(city_data, n_clusters))
    # Exibição de dados brutos
    if st.checkbox("Mostrar dados brutos cluster KMeans"):
        st.write(city_data)

    st.header("Clusterização de Acidentes por Cidade (Sparse K-Means)")
    city_data_sparse = preprocess_for_sparse_kmeans(df_filtered)
    fig_sparse, feature_weights = create_sparse_kmeans_map(city_data_sparse, n_clusters)
    st.plotly_chart(fig_sparse)
    st.write("Importância das características na clusterização:")
    st.write(feature_weights)

    # Exibição de dados brutos
    if st.checkbox("Mostrar dados brutos cluster Sparse KMeans"):
        st.write(city_data_sparse)

    st.header("Mapa de Acidentes")
    fig_map = create_accident_map(df_filtered)
    if fig_map:
        st.plotly_chart(fig_map)
    else:
        st.warning("Não há dados válidos para exibir no mapa.")

    # Visualizações clássicas
    st.header("Acidentes por Concessionária")
    st.plotly_chart(
        create_bar_chart(
            df_filtered,
            "Concessionaria",
            "Concessionaria",
            "Acidentes por Concessionária",
        )
    )

    st.header("Acidentes por UF")
    st.plotly_chart(create_bar_chart(df_filtered, "UF", "UF", "Acidentes por UF"))

    st.header("Acidentes ao Longo do Tempo")
    st.plotly_chart(create_time_series(df_filtered))

    st.header("Causas Diretas dos Acidentes")
    st.plotly_chart(
        create_pie_chart(df_filtered, "Causa_direta", "Distribuição das Causas Diretas")
    )

    st.header("Natureza dos Acidentes")
    st.plotly_chart(
        create_pie_chart(
            df_filtered, "Natureza", "Distribuição da Natureza dos Acidentes"
        )
    )

    if st.checkbox("Mostrar dados brutos"):
        st.subheader("Dados Brutos")
        st.write(df_filtered)


if __name__ == "__main__":
    main()
