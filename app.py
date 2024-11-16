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
        sep=",",
    )
    df["Data_Ocorrencia"] = pd.to_datetime(df["Data_Ocorrencia"], format="mixed")
    df["Quilometro_Inicial"] = pd.to_numeric(
        df["Quilometro_Inicial"].replace(",", "."), errors="coerce"
    )
    df["Hora_Ocorrencia"] = pd.to_datetime(
        df["Hora_Ocorrencia"], format="%H:%M"
    ).dt.time
    df["Mercadoria"] = df["Mercadoria"].fillna("Não Identificada")
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


# Funções de visualização
def create_cluster_map(city_data, n_clusters):
    scaler = StandardScaler()
    features = city_data[["num_acidentes", "Latitude", "Longitude"]]
    normalized_features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    city_data["cluster"] = kmeans.fit_predict(normalized_features)

    fig = px.scatter_mapbox(
        city_data,
        lat="Latitude",
        lon="Longitude",
        color="num_acidentes",
        size="num_acidentes",
        hover_name="Municipio",
        hover_data=["num_acidentes"],
        zoom=3,
        height=900,
        color_continuous_scale=px.colors.sequential.Oranges[
            3:
        ],  # Exclude lightest shades
    )
    fig.update_layout(
        mapbox_style="carto-darkmatter",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
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
        "Escolha o número de clusters", min_value=2, max_value=10, value=5
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
    st.header("Clusterização de Acidentes por Cidade")
    city_data = preprocess_for_kmeans(df_filtered)
    st.plotly_chart(create_cluster_map(city_data, n_clusters))

    # Exibição de dados brutos
    if st.checkbox("Mostrar dados brutos cluster"):
        st.write(city_data)

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
