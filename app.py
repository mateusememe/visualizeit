import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv(
        "datasets/acidents_ferroviarios_2004_2024_com_coords.csv",
        encoding="UTF-8",
        sep=",",
    )
    df["Data_Ocorrencia"] = pd.to_datetime(df["Data_Ocorrencia"], format="mixed")
    df["Quilômetro_Inicial"] = pd.to_numeric(
        df["Quilômetro_Inicial"].replace(",", "."), errors="coerce"
    )
    df["Hora_Ocorrencia"] = pd.to_datetime(
        df["Hora_Ocorrencia"], format="%H:%M"
    ).dt.time
    df["Mercadoria"] = df["Mercadoria"].fillna("Não Identificada")
    return df


df = load_data()

# Set page title
st.title("Visualização de Acidentes Ferroviários - ANTT")

# Sidebar for filtering
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

# Apply filters
if concessionaria:
    df = df[df["Concessionaria"].isin(concessionaria)]
if uf:
    df = df[df["UF"].isin(uf)]
if mercadoria:
    df = df[df["Mercadoria"].isin(mercadoria)]
df = df[
    (df["Data_Ocorrencia"].dt.date >= date_range[0])
    & (df["Data_Ocorrencia"].dt.date <= date_range[1])
]
df = df[
    (df["Hora_Ocorrencia"] >= time_range[0]) & (df["Hora_Ocorrencia"] <= time_range[1])
]

# Map visualization (moved to the beginning)
st.header("Mapa de Acidentes")
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
        height=300,
    )
    fig_map.update_layout(mapbox_style="open-street-map")
    fig_map.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    st.plotly_chart(fig_map)
else:
    st.warning("Não há dados válidos para exibir no mapa.")

# Other visualizations
st.header("Acidentes por Concessionária")
df_concessionaria = df["Concessionaria"].value_counts().reset_index()
df_concessionaria.columns = ["Concessionaria", "Numero_Acidentes"]
fig_concessionaria = px.bar(
    df_concessionaria,
    x="Concessionaria",
    y="Numero_Acidentes",
    labels={
        "Concessionaria": "Concessionária",
        "Numero_Acidentes": "Número de Acidentes",
    },
)
st.plotly_chart(fig_concessionaria)

st.header("Acidentes por UF")
df_uf = df["UF"].value_counts().reset_index()
df_uf.columns = ["UF", "Numero_Acidentes"]
fig_uf = px.bar(
    df_uf,
    x="UF",
    y="Numero_Acidentes",
    labels={"UF": "UF", "Numero_Acidentes": "Número de Acidentes"},
)
st.plotly_chart(fig_uf)

st.header("Acidentes ao Longo do Tempo")
df_time = df.groupby("Data_Ocorrencia").size().reset_index(name="Numero_Acidentes")
fig_time = px.line(
    df_time,
    x="Data_Ocorrencia",
    y="Numero_Acidentes",
    labels={"Data_Ocorrencia": "Data", "Numero_Acidentes": "Número de Acidentes"},
)
st.plotly_chart(fig_time)

st.header("Causas Diretas dos Acidentes")
df_causes = df["Causa_direta"].value_counts().reset_index()
df_causes.columns = ["Causa_direta", "Numero_Acidentes"]
fig_causes = px.pie(
    df_causes,
    values="Numero_Acidentes",
    names="Causa_direta",
    title="Distribuição das Causas Diretas",
)
st.plotly_chart(fig_causes)

st.header("Natureza dos Acidentes")
df_nature = df["Natureza"].value_counts().reset_index()
df_nature.columns = ["Natureza", "Numero_Acidentes"]
fig_nature = px.pie(
    df_nature,
    values="Numero_Acidentes",
    names="Natureza",
    title="Distribuição da Natureza dos Acidentes",
)
st.plotly_chart(fig_nature)

city_accidents = (
    df.groupby("Municipio")
    .agg(
        {
            "Latitude": "first",
            "Longitude": "first",
        }
    )
    .reset_index()
)

# Contagem separada de acidentes por município
accident_counts = df["Municipio"].value_counts().reset_index()
accident_counts.columns = ["Municipio", "num_acidentes"]

# Mesclagem dos dados de localização com a contagem de acidentes
city_accidents = city_accidents.merge(accident_counts, on="Municipio", how="left")
# Verificação e tratamento de valores nulos
city_accidents = city_accidents.dropna(subset=["num_acidentes"])


# def determine_optimal_clusters(data, max_clusters=10):
#     features = data[["num_acidentes"]]  # Modificado para usar apenas num_acidentes
#     scaler = StandardScaler()
#     normalized_features = scaler.fit_transform(features)

#     inertias = []
#     silhouette_scores = []

#     for k in range(2, max_clusters + 1):
#         kmeans = KMeans(n_clusters=k, random_state=42)
#         kmeans.fit(normalized_features)
#         inertias.append(kmeans.inertia_)
#         cluster_labels = kmeans.labels_
#         silhouette_scores.append(silhouette_score(normalized_features, cluster_labels))

#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

#     ax1.plot(range(2, max_clusters + 1), inertias, marker="o")
#     ax1.set_xlabel("Número de clusters")
#     ax1.set_ylabel("Inércia")
#     ax1.set_title("Método do Cotovelo")

#     ax2.plot(range(2, max_clusters + 1), silhouette_scores, marker="o")
#     ax2.set_xlabel("Número de clusters")
#     ax2.set_ylabel("Pontuação de Silhueta")
#     ax2.set_title("Método da Silhueta")

#     plt.tight_layout()
#     st.pyplot(fig)

#     return inertias, silhouette_scores


# inertias, silhouette_scores = determine_optimal_clusters(city_accidents)

# elbow_point = np.argmin(np.diff(inertias)) + 2
# silhouette_point = np.argmax(silhouette_scores) + 2

# st.write(f"Número recomendado de clusters pelo Método do Cotovelo: {elbow_point}")
# st.write(f"Número recomendado de clusters pelo Método da Silhueta: {silhouette_point}")


# Normalize os dados
scaler = StandardScaler()
features = city_accidents[["Latitude", "Longitude", "num_acidentes"]]
normalized_features = scaler.fit_transform(features)

# Realize a clusterização K-means
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
city_accidents["cluster"] = kmeans.fit_predict(normalized_features)

# Crie a visualização
fig = px.scatter_mapbox(
    city_accidents,
    lat="Latitude",
    lon="Longitude",
    color="cluster",
    size="num_acidentes",
    hover_name="Municipio",
    hover_data=["num_acidentes"],
    zoom=3,
    height=600,
    color_continuous_scale=px.colors.qualitative.Bold,
)

fig.update_layout(mapbox_style="open-street-map")

# Exiba o gráfico no Streamlit
st.subheader("Clusterização de Acidentes por Cidade")
st.plotly_chart(fig)

# Exiba os dados brutos se desejado
if st.checkbox("Mostrar dados brutos cluster"):
    st.write(city_accidents)

# Display raw data
if st.checkbox("Mostrar dados brutos"):
    st.subheader("Dados Brutos")
    st.write(df)
