"""
Railway Accidents Visualization Application
----------------------------------------
A Streamlit application for visualizing and analyzing railway accidents data from ANTT (Brazilian National Land Transportation Agency).
The app provides interactive visualizations including clustering analysis, maps, and comparative charts.

Author: Mateus Mendonça Monteiro
Version: 1.0.0
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from typing import Tuple, Dict, Optional, List, Any, NamedTuple
from dataclasses import dataclass

# Page Configuration
st.set_page_config(
    page_title="Railway Accidents Visualization - ANTT",
    layout="wide",
    initial_sidebar_state="expanded",
)


@dataclass
class ClusteringResult:
    """Data class to store clustering analysis results."""

    city_data: pd.DataFrame
    features: List[str]
    scaler: StandardScaler
    original_values: Dict[str, pd.Series]


class DataProcessor:
    """Handles data processing and feature engineering."""

    @staticmethod
    def preprocess_for_kmeans(df: pd.DataFrame) -> ClusteringResult:
        """
        Preprocesses data for K-means clustering analysis.

        Args:
            df (pd.DataFrame): Input DataFrame with railway accidents data.

        Returns:
            ClusteringResult: Processed data ready for clustering analysis.

        Note:
            - Filters data for specific date range
            - Aggregates data by city
            - Scales features for clustering
        """
        # Filter date range
        df = df[
            (df["Data_Ocorrencia"] >= "2020-12-01")
            & (df["Data_Ocorrencia"] <= "2024-07-31")
        ]

        # Aggregate city data
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

        # Store original values
        original_values = {
            "Latitude": city_data["Latitude"].copy(),
            "Longitude": city_data["Longitude"].copy(),
            "num_acidentes": city_data["num_acidentes"].copy(),
            "Interrupcao": city_data["Interrupcao"].copy(),
            "Prejuizo_Financeiro": city_data["Prejuizo_Financeiro"].copy(),
        }

        # Scale features
        features = ["num_acidentes", "Interrupcao", "Prejuizo_Financeiro"]
        scaler = StandardScaler()
        city_data[features] = scaler.fit_transform(city_data[features])

        return ClusteringResult(city_data, features, scaler, original_values)


class ConcessionaireAnalysis(NamedTuple):
    """Named tuple for storing concessionaire analysis results."""

    stacked_chart: go.Figure
    summary: pd.DataFrame


class DataLoader:
    """Handles data loading and preprocessing operations."""

    @staticmethod
    @st.cache_data
    def load_data() -> pd.DataFrame:
        """
        Loads and preprocesses the railway accidents dataset.

        Returns:
            pd.DataFrame: Preprocessed DataFrame containing railway accidents data.
        """

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


class SidebarFilters:
    """Handles sidebar filter creation and management."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize sidebar filters.

        Args:
            df (pd.DataFrame): Input DataFrame for filter options.
        """
        self.df = df

    def get_filters(self) -> Dict[str, Any]:
        """
        Creates and returns all sidebar filters.

        Returns:
            Dict[str, Any]: Dictionary containing all filter values.
        """
        st.sidebar.header("Filtros")

        filters = {
            "concessionaria": st.sidebar.multiselect(
                "Concessionária", options=self.df["Concessionaria"].unique()
            ),
            "uf": st.sidebar.multiselect("UF", options=self.df["UF"].unique()),
            "mercadoria": st.sidebar.multiselect(
                "Mercadoria", options=self.df["Mercadoria"].unique()
            ),
            "date_range": st.sidebar.date_input(
                "Intervalo de Data",
                [
                    self.df["Data_Ocorrencia"].min().date(),
                    self.df["Data_Ocorrencia"].max().date(),
                ],
                min_value=self.df["Data_Ocorrencia"].min().date(),
                max_value=self.df["Data_Ocorrencia"].max().date(),
            ),
            "time_range": st.sidebar.slider(
                "Intervalo de Hora",
                value=(
                    self.df["Hora_Ocorrencia"].min(),
                    self.df["Hora_Ocorrencia"].max(),
                ),
                format="HH:mm",
            ),
            "n_clusters": st.sidebar.slider(
                "Escolha o número de clusters", min_value=2, max_value=100, value=5
            ),
        }

        return filters


class DataFilter:
    """Handles data filtering operations."""

    @staticmethod
    def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """
        Applies all filters to the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame to filter.
            filters (Dict[str, Any]): Dictionary containing filter values.

        Returns:
            pd.DataFrame: Filtered DataFrame.
        """
        df_filtered = df.copy()

        if filters["concessionaria"]:
            df_filtered = df_filtered[
                df_filtered["Concessionaria"].isin(filters["concessionaria"])
            ]

        if filters["uf"]:
            df_filtered = df_filtered[df_filtered["UF"].isin(filters["uf"])]

        if filters["mercadoria"]:
            df_filtered = df_filtered[
                df_filtered["Mercadoria"].isin(filters["mercadoria"])
            ]

        df_filtered = df_filtered[
            (df_filtered["Data_Ocorrencia"].dt.date >= filters["date_range"][0])
            & (df_filtered["Data_Ocorrencia"].dt.date <= filters["date_range"][1])
        ]

        df_filtered = df_filtered[
            (df_filtered["Hora_Ocorrencia"] >= filters["time_range"][0])
            & (df_filtered["Hora_Ocorrencia"] <= filters["time_range"][1])
        ]

        return df_filtered


class Visualizer:
    """Handles creation of various visualizations."""

    @staticmethod
    def create_cluster_map(result: ClusteringResult, n_clusters: int) -> go.Figure:
        """
        Creates an interactive cluster map visualization.

        Args:
            result (ClusteringResult): Clustering analysis results.
            n_clusters (int): Number of clusters to create.

        Returns:
            go.Figure: Plotly figure object containing the cluster map.

        Note:
            - Uses K-means clustering
            - Sizes markers based on financial impact
            - Colors based on interruption time
        """
        features_array = result.city_data[result.features].values
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        result.city_data["cluster"] = kmeans.fit_predict(features_array)

        # Calculate centroids
        centroids_df = pd.DataFrame()
        for cluster in range(n_clusters):
            mask = result.city_data["cluster"] == cluster
            weight = result.original_values["num_acidentes"][mask]

            centroids_df.loc[cluster, "Latitude"] = np.average(
                result.original_values["Latitude"][mask], weights=weight
            )
            centroids_df.loc[cluster, "Longitude"] = np.average(
                result.original_values["Longitude"][mask], weights=weight
            )

        # Add cluster statistics
        cluster_stats = Visualizer._calculate_cluster_stats(
            result.city_data, result.original_values, n_clusters
        )
        centroids_df = centroids_df.join(cluster_stats)

        return Visualizer._create_plotly_map(centroids_df)

    @staticmethod
    def _calculate_cluster_stats(
        city_data: pd.DataFrame, original_values: Dict[str, pd.Series], n_clusters: int
    ) -> pd.DataFrame:
        """
        Calculates statistics for each cluster.

        Args:
            city_data (pd.DataFrame): Processed city data with cluster assignments.
            original_values (Dict[str, pd.Series]): Original (unscaled) values.
            n_clusters (int): Number of clusters.

        Returns:
            pd.DataFrame: DataFrame containing cluster statistics.
        """
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

        return cluster_stats

    @staticmethod
    def _create_plotly_map(centroids_df: pd.DataFrame) -> go.Figure:
        """
        Creates the Plotly map visualization.

        Args:
            centroids_df (pd.DataFrame): DataFrame containing centroid information.

        Returns:
            go.Figure: Plotly figure object containing the map visualization.
        """
        # Scale marker sizes
        min_size, max_size = 20, 70
        if centroids_df["prejuizo_total"].max() != centroids_df["prejuizo_total"].min():
            centroids_df["marker_size"] = (
                centroids_df["prejuizo_total"] - centroids_df["prejuizo_total"].min()
            ) / (
                centroids_df["prejuizo_total"].max()
                - centroids_df["prejuizo_total"].min()
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

        map_figure = px.scatter_mapbox(
            centroids_df,
            lat="Latitude",
            lon="Longitude",
            size="marker_size",
            color="total_interrupcao",
            custom_data=[
                "prejuizo_hover",
                "principais_cidades",
                "total_acidentes",
                "prejuizo_total",
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
        ).update_layout(
            mapbox_style="carto-darkmatter",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        # Update marker appearance
        map_figure.update_traces(
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

        map_figure.update_layout(
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

        return map_figure

    @staticmethod
    def create_accident_map(df: pd.DataFrame) -> Optional[go.Figure]:
        """
        Creates a map visualization of accident locations.

        Args:
            df (pd.DataFrame): DataFrame containing accident data.

        Returns:
            Optional[go.Figure]: Plotly figure object containing the map visualization,
                               or None if no valid data.
        """
        df_map = df.dropna(subset=["Municipio", "Latitude", "Longitude"])

        if df_map.empty:
            return None

        return px.scatter_mapbox(
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
        ).update_layout(
            mapbox_style="open-street-map", margin={"r": 0, "t": 0, "l": 0, "b": 0}
        )

    @staticmethod
    def create_bar_chart(df: pd.DataFrame, x: str, title: str) -> go.Figure:
        """
        Creates a bar chart visualization.

        Args:
            df (pd.DataFrame): Input DataFrame.
            x (str): Column name for x-axis.
            title (str): Chart title.

        Returns:
            go.Figure: Plotly figure object containing the bar chart.
        """
        financial_by_company = df.groupby(x)["Prejuizo_Financeiro"].sum().reset_index()
        financial_by_company.columns = [x, "Prejuízo Total"]

        financial_by_company["Prejuízo Formatado"] = financial_by_company[
            "Prejuízo Total"
        ].apply(
            lambda x: f"R$ {x:,.2f}".replace(",", "X")
            .replace(".", ",")
            .replace("X", ".")
        )

        return (
            px.bar(
                financial_by_company,
                x=x,
                y="Prejuízo Total",
                title=title,
                text="Prejuízo Formatado",
            )
            .update_layout(
                xaxis_title=x, yaxis_title="Prejuízo Financeiro (R$)", bargap=0.2
            )
            .update_traces(textposition="auto")
        )

    @staticmethod
    def create_concessionaria_analysis(df: pd.DataFrame) -> ConcessionaireAnalysis:
        """
        Creates comprehensive analysis of concessionaires.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            ConcessionaireAnalysis: Named tuple containing stacked chart and summary DataFrame.
        """
        # Group data by Concessionaria
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

        raw_values = grouped.copy()
        grouped["color_index"] = range(len(grouped))

        # Create normalized values
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

        # Create summary
        summary = raw_values.copy()
        summary["Prejuizo_Formatado"] = summary["Prejuizo_Financeiro"].apply(
            lambda x: f"R$ {x:,.2f}".replace(",", "X")
            .replace(".", ",")
            .replace("X", ".")
        )
        summary["Interrupcao_Formatada"] = summary["Interrupcao"].apply(
            lambda x: f"{x:,.1f}h"
        )
        summary["N_Acidentes"] = summary["Data_Ocorrencia"]

        # Calculate scores
        summary["Score"] = (
            grouped["Prejuizo_Financeiro_norm"]
            + grouped["Interrupcao_norm"]
            + grouped["Data_Ocorrencia_norm"]
        ) / 3

        summary = summary.sort_values("Score")

        return ConcessionaireAnalysis(
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


def show_accident_map(visualizer: Visualizer, df_filtered: pd.DataFrame):
    st.header("Mapa de Acidentes")
    accident_map = visualizer.create_accident_map(df_filtered)
    if accident_map:
        st.plotly_chart(accident_map)
    else:
        st.warning("Não há dados válidos para exibir no mapa.")


def show_concessionaria_analysis(visualizer: Visualizer, df_filtered: pd.DataFrame):
    st.header("Acidentes por Concessionária")
    company_chart = visualizer.create_bar_chart(
        df_filtered, "Concessionaria", "Distribuição de Acidentes por Concessionária"
    )
    st.plotly_chart(company_chart)

    # Display comparative analysis
    st.header("Análise Comparativa de Concessionárias")
    analysis = visualizer.create_concessionaria_analysis(df_filtered)

    st.subheader("Distribuição Normalizada de Métricas")
    st.markdown(
        """
    - Valores normalizados permitem comparação justa entre métricas
    - Altura total das barras indica impacto total
    """
    )
    st.plotly_chart(analysis.stacked_chart, use_container_width=True)

    st.subheader("Resumo por Concessionária")
    st.markdown(
        """
    - Valores absolutos formatados
    - Score composto (média das métricas normalizadas)
    - Ordem do melhor para o pior desempenho geral
    """
    )
    st.dataframe(
        analysis.summary.style.background_gradient(subset=["Score"], cmap="RdYlGn_r")
    )


def tabs_handler(
    visualizer: Visualizer, df_filtered: pd.DataFrame, filters: Dict
) -> None:
    kmeans_categorical, kmeans_numerical, accident_map_tab, concessionaria_analysis = (
        st.tabs(
            [
                "K-Means Categórico",
                "K-Means Numérico",
                "Mapa de Acidentes - Distribuição Geográfica",
                "Acidentes por Concessionária",
            ]
        )
    )

    with kmeans_categorical:
        st.header("Clusterização de Acidentes por Cidade (K-Means Numérico)")
        clustering_result = DataProcessor.preprocess_for_kmeans(df_filtered)
        cluster_map = visualizer.create_cluster_map(
            clustering_result, filters["n_clusters"]
        )
        st.plotly_chart(cluster_map)

        if st.checkbox("Mostrar dados brutos cluster KMeans"):
            st.write(clustering_result.city_data)
    with kmeans_numerical:
        st.header("Clusterização de Acidentes por Cidade (K-Means Categórico)")
        clustering_result = DataProcessor.preprocess_for_kmeans(df_filtered)
        cluster_map = visualizer.create_cluster_map(
            clustering_result, filters["n_clusters"]
        )
        st.plotly_chart(cluster_map)

        if st.checkbox("Mostrar dados brutos cluster KMeans"):
            st.write(clustering_result.city_data)

    with accident_map_tab:
        show_accident_map(visualizer, df_filtered)

    with concessionaria_analysis:
        show_concessionaria_analysis(visualizer, df_filtered)


def main():
    """Main application entry point."""
    st.title("Visualização de Acidentes Ferroviários - ANTT")

    # Load data
    df = DataLoader.load_data()

    # Create sidebar filters
    sidebar = SidebarFilters(df)
    filters = sidebar.get_filters()

    # Apply filters
    df_filtered = DataFilter.apply_filters(df, filters)

    # Create visualizations object handler
    visualizer = Visualizer()

    # Display clustering analysis
    tabs_handler(visualizer, df_filtered, filters)


if __name__ == "__main__":
    main()
