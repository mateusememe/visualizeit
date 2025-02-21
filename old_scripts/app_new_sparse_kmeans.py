import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

from models.SparseKmeans import SparseKMeans

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
    df["Prejuizo_Financeiro"] = pd.to_numeric(
        df["Prejuizo_Financeiro"].str.replace(",", "."), errors="coerce"
    )
    df["Prejuizo_Financeiro"] = df["Prejuizo_Financeiro"].fillna(0)

    return df


def preprocess_features(df):
    """Pré-processa as características numéricas e categóricas."""
    # Características numéricas
    numeric_features = [
        "Interrupcao",
        # "N_feridos",
        # "N_obitos",
        "Prejuizo_Financeiro",
        "Latitude",
        "Longitude",
    ]

    # Características categóricas
    # categorical_features = [
    # "Gravidade",
    # "Concessionaria",
    # "UF",
    # "Linha",
    # "Perímetro_Urbano",
    # "Causa_direta",
    # "Natureza",
    # "Servico_Transporte",
    # "Mercadoria",
    # "PN",
    # ]

    # Criar cópia do DataFrame
    df_processed = df.copy()

    # Imputar valores ausentes em características numéricas
    imputer = SimpleImputer(strategy="mean")
    df_processed[numeric_features] = imputer.fit_transform(
        df_processed[numeric_features]
    )

    # Normalizar todas as características
    scaler = StandardScaler()
    features = numeric_features
    df_processed[features] = scaler.fit_transform(df_processed[features])
    # df_processed['Prejuizo_Financeiro'] = df['Prejuizo_Financeiro']
    return df_processed, features, scaler


@st.cache_data
def preprocess_for_kmeans(df):
    """Pré-processa dados para K-means regular."""

    # Aggregate data by city first
    city_data = (
        df.groupby("Municipio")
        .agg(
            {
                "Latitude": "first",
                "Longitude": "first",
                "Data_Ocorrencia": "count",
                "Interrupcao": "sum",
                "Prejuizo_Financeiro": "sum",
            }
        )
        .reset_index()
    )

    city_data.rename(columns={"Data_Ocorrencia": "num_acidentes"}, inplace=True)

    # Store original values before scaling
    original_values = {
        "Latitude": city_data["Latitude"].copy(),
        "Longitude": city_data["Longitude"].copy(),
        "num_acidentes": city_data["num_acidentes"].copy(),
        "Interrupcao": city_data["Interrupcao"].copy(),
        "Prejuizo_Financeiro": city_data["Prejuizo_Financeiro"].copy(),
    }

    # Scale only the features we want to use for clustering
    scaler = StandardScaler()
    features = ["num_acidentes", "Interrupcao", "Prejuizo_Financeiro"]
    city_data[features] = scaler.fit_transform(city_data[features])

    return city_data, features, scaler, original_values


def create_cluster_map(city_data, features, n_clusters, scaler, original_values):
    """Cria o mapa de clusters com os dados corrigidos."""
    # Use only scaled features for clustering
    features_array = city_data[features].values

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    city_data["cluster"] = kmeans.fit_predict(features_array)

    # Create centroids DataFrame with correct geographical coordinates
    centroids_df = pd.DataFrame()

    # Calculate weighted average of lat/long for each cluster
    for cluster in range(n_clusters):
        mask = city_data["cluster"] == cluster
        weight = original_values["num_acidentes"][mask]

        centroids_df.loc[cluster, "Latitude"] = np.average(
            original_values["Latitude"][mask], weights=weight
        )
        centroids_df.loc[cluster, "Longitude"] = np.average(
            original_values["Longitude"][mask], weights=weight
        )

    # Add cluster statistics using original (unscaled) values
    cluster_stats = pd.DataFrame()
    for cluster in range(n_clusters):
        mask = city_data["cluster"] == cluster
        cluster_stats.loc[cluster, "total_acidentes"] = original_values[
            "num_acidentes"
        ][mask].sum()
        cluster_stats.loc[cluster, "total_interrupcao"] = original_values[
            "Interrupcao"
        ][mask].sum()
        cluster_stats.loc[cluster, "prejuizo_total"] = original_values[
            "Prejuizo_Financeiro"
        ][mask].sum()
        cluster_stats.loc[cluster, "principais_cidades"] = ", ".join(
            city_data.loc[mask, "Municipio"].head(3)
        )

    # Merge statistics with centroids
    centroids_df = centroids_df.join(cluster_stats)

    # Scale the size values to be more visible
    min_size = 20  # Minimum marker size
    max_size = 50  # Maximum marker size

    # Create a size scale based on prejuizo_total
    if centroids_df["prejuizo_total"].max() != centroids_df["prejuizo_total"].min():
        centroids_df["marker_size"] = (
            centroids_df["prejuizo_total"] - centroids_df["prejuizo_total"].min()
        ) / (
            centroids_df["prejuizo_total"].max() - centroids_df["prejuizo_total"].min()
        ) * (
            max_size - min_size
        ) + min_size
    else:
        centroids_df["marker_size"] = min_size

    # Format currency for hover display
    centroids_df["prejuizo_hover"] = centroids_df["prejuizo_total"].apply(
        lambda valor: f"R$ {valor:,.2f}".replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )

    # Create the visualization with adjusted size parameters
    fig = px.scatter_mapbox(
        centroids_df,
        lat="Latitude",
        lon="Longitude",
        size="marker_size",  # Use our scaled size column
        color="total_interrupcao",
        custom_data=[
            "prejuizo_hover",
            "principais_cidades",
            "total_acidentes",
            "prejuizo_total",  # Added for reference
        ],
        zoom=4,
        height=900,
        title="Centroides dos Clusters (K-Means)",
        color_continuous_scale=px.colors.sequential.Oranges[1:],
        labels={
            "prejuizo_total": "Prejuízo Total (R$)",
            "total_interrupcao": "Tempo Total de Interrupção (horas)",
            "total_acidentes": "Total de Acidentes",
            "principais_cidades": "Principais Cidades",
        },
    )

    # Update marker appearance
    fig.update_traces(
        marker=dict(
            sizemode="diameter",  # Use diameter mode for better size scaling
            sizeref=1,  # Adjust this value to change the overall size scale
            sizemin=min_size / 2,  # Minimum size to ensure visibility
        ),
        hovertemplate="<br>".join(
            [
                "Latitude: %{lat:.4f}",
                "Longitude: %{lon:.4f}",
                "Total de Acidentes: %{customdata[2]:,.0f}",
                "Tempo de Interrupção: %{marker.color:.1f} horas",
                "Prejuízo Total: %{customdata[0]}",
                "Principais Cidades: %{customdata[1]}",
                "<extra></extra>",
            ]
        ),
    )

    fig.update_layout(
        mapbox_style="carto-darkmatter",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title={"y": 0.98, "x": 0.5, "xanchor": "center", "yanchor": "top"},
        mapbox=dict(
            zoom=4,
        ),
        annotations=[
            dict(
                x=0.02,
                y=0.98,
                xref="paper",
                yref="paper",
                text="<b>Legenda:</b><br>"
                + "• Tamanho dos círculos: Prejuízo financeiro total<br>"
                + "• Cor dos círculos: Tempo total de interrupção",
                showarrow=False,
                font=dict(size=12, color="white"),
                align="left",
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor="white",
                borderwidth=1,
                borderpad=4,
                xanchor="left",
                yanchor="top",
            )
        ],
    )

    return fig


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
    # Create sum of financial losses by company
    financial_by_company = df.groupby(x)["Prejuizo_Financeiro"].sum().reset_index()
    financial_by_company.columns = [x, "Prejuízo Total"]

    # Format the values to Brazilian currency
    financial_by_company["Prejuízo Formatado"] = financial_by_company[
        "Prejuízo Total"
    ].apply(
        lambda x: f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    )

    fig = px.bar(
        financial_by_company,
        x=x,
        y="Prejuízo Total",
        title=title,
        text="Prejuízo Formatado",  # Show formatted values on bars
    )

    fig.update_layout(xaxis_title=x, yaxis_title="Prejuízo Financeiro (R$)", bargap=0.2)

    # Update text position to be inside or outside bars depending on value
    fig.update_traces(textposition="auto")

    return fig


def create_concessionaria_analysis(df):
    grouped = (
        df.groupby("Concessionaria")
        .agg(
            {
                "Prejuizo_Financeiro": "sum",
                "Interrupcao": "sum",
                "Data_Ocorrencia": "count",
            }
        )
        .reset_index()
    )

    # Store raw values before normalization
    raw_values = grouped.copy()

    # Create a numeric index for each Concessionaria for coloring
    grouped["color_index"] = range(len(grouped))

    # Create normalized values for stacked bar chart
    for column in ["Prejuizo_Financeiro", "Interrupcao", "Data_Ocorrencia"]:
        grouped[f"{column}_norm"] = grouped[column] / grouped[column].max()

    # Create stacked bar chart
    fig_stacked = px.bar(
        grouped,
        x="Concessionaria",
        y=["Prejuizo_Financeiro_norm", "Interrupcao_norm", "Data_Ocorrencia_norm"],
        title="Distribuição de Métricas por Concessionária (Normalizado)",
        labels={
            "value": "Valor Normalizado",
            "variable": "Métrica",
            "Concessionaria": "Concessionária",
            "Prejuizo_Financeiro_norm": "Prejuízo Financeiro",
            "Interrupcao_norm": "Tempo de Interrupção",
            "Data_Ocorrencia_norm": "Número de Acidentes",
        },
        barmode="relative",
    )

    # Create summary table
    summary = raw_values.copy()
    summary["Prejuizo_Formatado"] = summary["Prejuizo_Financeiro"].apply(
        lambda x: f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    )
    summary["Interrupcao_Formatada"] = summary["Interrupcao"].apply(
        lambda x: f"{x:,.1f}h"
    )
    summary["N_Acidentes"] = summary["Data_Ocorrencia"]

    # Calculate scores (lower is better)
    summary["Score"] = (
        grouped["Prejuizo_Financeiro_norm"]
        + grouped["Interrupcao_norm"]
        + grouped["Data_Ocorrencia_norm"]
    ) / 3

    summary = summary.sort_values("Score")

    return (
        fig_stacked,
        summary[
            [
                "Concessionaria",
                "Prejuizo_Formatado",
                "Interrupcao_Formatada",
                "N_Acidentes",
                "Score",
            ]
        ],
    )


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
    city_data, features, scaler, original_values = preprocess_for_kmeans(df_filtered)
    st.plotly_chart(
        create_cluster_map(city_data, features, n_clusters, scaler, original_values)
    )
    if st.checkbox("Mostrar dados brutos cluster KMeans"):
        st.write(city_data)

    # st.header("Clusterização de Acidentes por Cidade (Sparse K-Means)")
    # city_data_sparse, features_sparse, scaler_sparse = preprocess_for_sparse_kmeans(
    #     df_filtered
    # )
    # fig_sparse, feature_weights = create_sparse_kmeans_map(
    #     city_data_sparse, features_sparse, n_clusters, scaler_sparse
    # )
    # st.plotly_chart(fig_sparse)
    # st.write("Importância das características na clusterização:")
    # st.write(feature_weights)
    # if st.checkbox("Mostrar dados brutos cluster Sparse KMeans"):
    #     st.write(city_data_sparse)

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
            None,  # Não precisamos mais do y, pois calculamos dentro da função
            "Distribuição de Acidentes por Concessionária",
        )
    )

    st.header("Análise Comparativa de Concessionárias")

    fig_stacked, summary = create_concessionaria_analysis(df_filtered)

    # Show stacked bar chart with explanation
    st.subheader("Distribuição Normalizada de Métricas")
    st.markdown(
        """
    Este gráfico mostra a contribuição relativa de cada métrica normalizada:
    - Valores normalizados permitem comparação justa entre métricas
    - Altura total das barras indica impacto total
    """
    )
    st.plotly_chart(fig_stacked, use_container_width=True)

    # Show summary table with explanation
    st.subheader("Resumo por Concessionária")
    st.markdown(
        """
    Esta tabela apresenta um resumo com:
    - Valores absolutos formatados
    - Score composto (média das métricas normalizadas)
    - Ordem do melhor para o pior desempenho geral
    """
    )
    st.dataframe(summary.style.background_gradient(subset=["Score"], cmap="RdYlGn_r"))


if __name__ == "__main__":
    main()
