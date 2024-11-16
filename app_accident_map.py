import streamlit as st
import pandas as pd
import plotly.express as px


@st.cache_data
def load_data():
    df = pd.read_csv(
        "datasets/acidents_ferroviarios_2004_2024_com_coords.csv",
        encoding="UTF-8",
        sep=";",
    )
    df["Data_Ocorrencia"] = pd.to_datetime(df["Data_Ocorrencia"], format="mixed")
    return df


def create_accident_map(df, year_groups, colors):
    df_map = df.dropna(subset=["Municipio", "Latitude", "Longitude", "Data_Ocorrencia"])

    if not df_map.empty:
        # Create year groups
        df_map["Ano"] = df_map["Data_Ocorrencia"].dt.year
        labels = [f"{start}-{end}" for start, end in year_groups]
        bins = [year_group[0] for year_group in year_groups] + [year_groups[-1][1] + 1]
        df_map["Ano_Grupo"] = pd.cut(df_map["Ano"], bins=bins, labels=labels)

        # Criar um dicionário de cores personalizado
        color_discrete_map = {label: color for label, color in zip(labels, colors)}

        # Plot the map with colors by year group
        fig_map = px.scatter_mapbox(
            df_map,
            lat="Latitude",
            lon="Longitude",
            color="Ano_Grupo",
            color_discrete_map=color_discrete_map,  # Usar o mapa de cores personalizado
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
        )
        fig_map.update_layout(mapbox_style="open-street-map")
        fig_map.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        return fig_map
    return None


def main():
    st.title("Mapa de Acidentes Ferroviários")

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

    # Sidebar para configuração dos grupos de ano e cores
    st.sidebar.header("Configuração de Legenda Grupo-Ano")

    num_groups = st.sidebar.number_input(
        "Número de Grupos de Anos", min_value=2, max_value=10, value=3, step=1
    )
    year_groups = []
    colors = []

    for i in range(num_groups):
        with st.sidebar.expander(f"Grupo {i+1}"):
            start_year = st.number_input(
                f"Ano Inicial do Grupo {i+1}",
                min_value=2004,
                max_value=2023,
                value=2004 + (i * 3),
                step=1,
            )
            end_year = st.number_input(
                f"Ano Final do Grupo {i+1}",
                min_value=2005,
                max_value=2024,
                value=2007 + (i * 3),
                step=1,
            )
            color = st.color_picker(f"Cor do Grupo {i+1}", "#FF0000", key=f"color_{i}")
            year_groups.append((start_year, end_year))
            colors.append(color)

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

    # Exibição do mapa de acidentes
    fig_map = create_accident_map(df_filtered, year_groups, colors)
    if fig_map:
        st.plotly_chart(fig_map)
    else:
        st.warning("Não há dados válidos para exibir no mapa.")


if __name__ == "__main__":
    main()
