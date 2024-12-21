import logging
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, Optional
from plotly.subplots import make_subplots


from models.robust_sparse_kmeans import SparseKMeansResult
from utils.data import (
    CategoricalClusteringResult,
    ClusteringResult,
    ConcessionaireAnalysis,
)


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
            color_continuous_scale=px.colors.sequential.Redor[1:],
            labels={
                "prejuizo_total": "Prejuízo Total (R$)",
                "total_interrupcao": "Tempo Total de Interrupção (horas)",
                "total_acidentes": "Total de Acidentes",
                "principais_cidades": "Principais Cidades",
            },
        ).update_layout(
            mapbox_style="open-street-map",
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
            mapbox_style="open-street-map",
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
        Creates a map visualization of accident locations with white theme.

        Args:
            df (pd.DataFrame): DataFrame containing accident data.

        Returns:
            Optional[go.Figure]: Plotly figure object containing the map visualization,
                            or None if no valid data.
        """
        df_map = df.dropna(subset=["Municipio", "Latitude", "Longitude"])

        if df_map.empty:
            return None

        # Format date for hover display
        df_map["Data_Formatada"] = pd.to_datetime(
            df_map["Data_Ocorrencia"]
        ).dt.strftime("%d/%m/%Y")

        # Create the map with enhanced features
        fig = px.scatter_mapbox(
            df_map,
            lat="Latitude",
            lon="Longitude",
            hover_name="Municipio",
            hover_data={
                "Latitude": ":.4f",
                "Longitude": ":.4f",
                "Concessionaria": True,
                "Data_Formatada": True,
                "Linha": True,
                "Mercadoria": True,
            },
            zoom=4,
            height=800,
            color_discrete_sequence=["red"],
        )

        # Update layout for better visualization
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",  # White theme map
                center=dict(lat=-15.77, lon=-47.92),
            ),
            margin={"r": 0, "t": 0, "l": 0, "b": 0},  # No margins
            width=None,  # Allow responsive width
            showlegend=False,
        )

        # Update hover template for better information display
        fig.update_traces(
            hovertemplate=(
                "<b>%{hovertext}</b><br><br>"
                + "Latitude: %{lat:.4f}<br>"
                + "Longitude: %{lon:.4f}<br>"
                + "Concessionária: %{customdata[2]}<br>"
                + "Data: %{customdata[3]}<br>"
                + "Linha: %{customdata[4]}<br>"
                + "Mercadoria: %{customdata[5]}<br>"
                + "<extra></extra>"
            )
        )

        return fig

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

        # Adicionar valores formatados para o hover
        grouped["Prejuizo_Hover"] = grouped["Prejuizo_Financeiro"].apply(
            lambda x: f"R$ {x:,.2f}".replace(",", "X")
            .replace(".", ",")
            .replace("X", ".")
        )
        grouped["Interrupcao_Hover"] = grouped["Interrupcao"].apply(
            lambda x: f"{x:,.1f} horas"
        )
        grouped["Acidentes_Hover"] = grouped["Data_Ocorrencia"].apply(
            lambda x: f"{x:,} acidentes".replace(",", ".")
        )

        # Create stacked bar chart with enhanced customization
        fig_stacked = px.bar(
            grouped,
            x="Concessionaria",
            y=["Prejuizo_Financeiro_norm", "Interrupcao_norm", "Data_Ocorrencia_norm"],
            title="Análise Comparativa de Métricas por Concessionária",
            labels={
                "value": "Valor Normalizado (0-1)",
                "variable": "Métrica Analisada",
                "Concessionaria": "Concessionária",
                "Prejuizo_Financeiro_norm": "Prejuízo Financeiro",
                "Interrupcao_norm": "Tempo de Interrupção",
                "Data_Ocorrencia_norm": "Número de Acidentes",
            },
            barmode="relative",
            color_discrete_sequence=[
                "rgb(99,110,250)",
                "rgb(239,85,59)",
                "rgb(0,204,150)",
            ],
            custom_data=[
                grouped["Prejuizo_Hover"],
                grouped["Interrupcao_Hover"],
                grouped["Acidentes_Hover"],
            ],
        )

        # Enhance the figure layout
        fig_stacked.update_layout(
            # title={
            #     "text": "Análise Comparativa de Métricas por Concessionária",
            #     "y": 0.95,
            #     "x": 0.5,
            #     "xanchor": "right",
            #     "yanchor": "top",
            #     "font": {"size": 24},
            # },
            template="plotly_white",
            showlegend=True,
            legend={
                "title": "Métricas Analisadas",
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "right",
                "x": 1,
            },
            xaxis={
                "title": "Concessionária",
                "tickangle": -45,
                "title_font": {"size": 14},
            },
            yaxis={
                "title": "Valor Normalizado (0-1)",
                "title_font": {"size": 14},
            },
            margin=dict(t=120, b=100),
            height=700,
        )

        # Update hover template
        fig_stacked.update_traces(
            hovertemplate=(
                "<b>%{x}</b><br>"
                + "Prejuízo Financeiro: %{customdata[0]}<br>"
                + "Tempo de Interrupção: %{customdata[1]}<br>"
                + "Número de Acidentes: %{customdata[2]}<br>"
                + "Valor Normalizado: %{y:.3f}<br>"
                + "<extra></extra>"
            )
        )

        # Add explanatory annotation
        fig_stacked.add_annotation(
            text=(
                "Valores normalizados (0-1) para comparação entre métricas.<br>"
                + "Passe o mouse sobre as barras para ver os valores reais."
            ),
            xref="paper",
            yref="paper",
            x=0,
            y=1.1,
            showarrow=False,
            font=dict(size=12, color="gray"),
            align="left",
        )

        # Create summary with enhanced formatting
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
        summary["Índice de Performance"] = (
            grouped["Prejuizo_Financeiro_norm"]
            + grouped["Interrupcao_norm"]
            + grouped["Data_Ocorrencia_norm"]
        ) / 3

        summary = summary.sort_values("Índice de Performance")

        # Prepare final summary with renamed columns
        final_summary = summary[
            [
                "Concessionaria",
                "Prejuizo_Formatado",
                "Interrupcao_Formatada",
                "N_Acidentes",
                "Índice de Performance",
            ]
        ].rename(
            columns={
                "Concessionaria": "Concessionária",
                "Prejuizo_Formatado": "Prejuízo Financeiro",
                "Interrupcao_Formatada": "Tempo de Interrupção",
                "N_Acidentes": "Número de Acidentes",
            }
        )

        return ConcessionaireAnalysis(fig_stacked, final_summary)

    def show_accident_map(self, df_filtered: pd.DataFrame):
        st.header("Mapa de Acidentes")

        accident_map = self.create_accident_map(df_filtered)
        if accident_map:
            # Usando container wide para maximizar largura
            with st.container():
                st.plotly_chart(accident_map, use_container_width=True)
        else:
            st.warning("Não há dados válidos para exibir no mapa.")

    @staticmethod
    def show_concessionaria_analysis(df: pd.DataFrame):
        """
        Shows the concessionaire analysis visualization.

        Args:
            df (pd.DataFrame): Input DataFrame
        """
        st.header("Acidentes por Concessionária")
        analysis = Visualizer.create_concessionaria_analysis(df)

        # Plot the stacked chart
        st.plotly_chart(analysis.stacked_chart, use_container_width=True)

        # Display comparative analysis with correct column styling
        st.subheader("Resumo por Concessionária")
        st.markdown(
            """
        - Valores absolutos formatados
        - Ordem do melhor para o pior desempenho
        """
        )
        # Aplicar estilo apenas na coluna 'Índice de Performance'
        st.dataframe(
            analysis.summary.style.background_gradient(
                subset=["Índice de Performance"], cmap="RdYlGn_r"
            )
        )

    @staticmethod
    def create_categorical_cluster_map(
        result: CategoricalClusteringResult, n_clusters: int
    ) -> go.Figure:
        """
        Creates an interactive cluster map visualization for categorical clustering.

        Args:
            result (CategoricalClusteringResult): Clustering analysis results
            n_clusters (int): Number of clusters to create

        Returns:
            go.Figure: Plotly figure object containing the cluster map
        """
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(result.city_data[result.features])

        # Calculate centroids for each cluster
        centroids_df = pd.DataFrame()
        for cluster in range(n_clusters):
            mask = clusters == cluster
            # Weight by number of accidents in each city
            weight = result.original_values["accident_count"][mask]

            centroids_df.loc[cluster, "Latitude"] = np.average(
                result.original_values["Latitude"][mask], weights=weight
            )
            centroids_df.loc[cluster, "Longitude"] = np.average(
                result.original_values["Longitude"][mask], weights=weight
            )

            # Add cluster statistics
            centroids_df.loc[cluster, "size"] = mask.sum()
            centroids_df.loc[cluster, "total_acidentes"] = weight.sum()

            # Get top features for this cluster
            cluster_center = kmeans.cluster_centers_[cluster]
            top_features_idx = np.argsort(cluster_center)[-3:]  # Get top 3 features
            top_features = [result.feature_names[i] for i in top_features_idx]
            centroids_df.loc[cluster, "characteristic_features"] = ", ".join(
                top_features
            )

        min_size, max_size = 20, 100
        if (
            centroids_df["total_acidentes"].max()
            != centroids_df["total_acidentes"].min()
        ):
            centroids_df["marker_size"] = (
                centroids_df["total_acidentes"] - centroids_df["total_acidentes"].min()
            ) / (
                centroids_df["total_acidentes"].max()
                - centroids_df["total_acidentes"].min()
            ) * (
                max_size - min_size
            ) + min_size
        else:
            centroids_df["marker_size"] = min_size

        # Create the map visualization
        map_figure = px.scatter_mapbox(
            centroids_df,
            lat="Latitude",
            lon="Longitude",
            size="marker_size",
            color=centroids_df.index,
            custom_data=["size", "total_acidentes", "characteristic_features"],
            zoom=4,
            height=900,
            title="Centroides dos Clusters Categóricos (K-Means)",
            color_continuous_scale=px.colors.sequential.Redor[1:],
        )

        map_figure.update_layout(
            mapbox_style="open-street-map",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        # Update hover template
        map_figure.update_traces(
            hovertemplate=(
                "<br>".join(
                    [
                        "Latitude: %{lat:.4f}",
                        "Longitude: %{lon:.4f}",
                        "Cidades no cluster: %{customdata[0]}",
                        "Total de acidentes: %{customdata[1]}",
                        "Características principais:<br>%{customdata[2]}",
                        "<extra></extra>",
                    ]
                )
            )
        )

        return map_figure


class ClusteringAnalyzer:
    """
    Handles analysis and visualization of categorical clustering results.
    Includes time period processing.
    """

    PERIOD_LABELS = {
        1: "00:00-06:00",
        2: "06:00-12:00",
        3: "12:00-18:00",
        4: "18:00-00:00",
    }

    def __init__(
        self,
        df: pd.DataFrame,
        cluster_labels: np.ndarray,
        n_clusters: int,
        city_data: pd.DataFrame,
    ):
        """
        Initialize the analyzer with clustering results and process time periods.

        Args:
            df (pd.DataFrame): Original non-aggregated DataFrame
            cluster_labels (np.ndarray): Array of cluster assignments for cities
            n_clusters (int): Number of clusters used
            city_data (pd.DataFrame): Aggregated city data used for clustering
        """
        self.df = df.copy()
        self.city_data = city_data.copy()
        self.n_clusters = n_clusters

        # Process time periods
        self.df["Periodo"] = self.df["Hora_Ocorrencia"].apply(self._assign_period)

        # Add cluster assignments to city_data
        self.city_data["Cluster"] = cluster_labels

        # Map clusters back to original data using municipality
        cluster_map = self.city_data[["Municipio", "Cluster"]].set_index("Municipio")[
            "Cluster"
        ]
        self.df["Cluster"] = self.df["Municipio"].map(cluster_map)

        # Define a consistent color sequence
        self.color_sequence = px.colors.qualitative.Set3[:n_clusters]

    def _assign_period(self, time_str: str) -> str:
        """
        Convert time string to period label.

        Args:
            time_str: Time in HH:MM format

        Returns:
            str: Period label
        """
        try:
            hour = int(str(time_str).split(":")[0])
            if 0 <= hour < 6:
                return self.PERIOD_LABELS[1]
            elif 6 <= hour < 12:
                return self.PERIOD_LABELS[2]
            elif 12 <= hour < 18:
                return self.PERIOD_LABELS[3]
            else:
                return self.PERIOD_LABELS[4]
        except (ValueError, AttributeError, IndexError):
            return self.PERIOD_LABELS[1]  # Default to first period if error

    def create_geographical_cluster_map(self) -> go.Figure:
        """
        Creates a geographical visualization of clusters using actual coordinates.
        Uses direct coordinate values from the dataset to ensure accurate positioning
        within Brazil's territory.
        """
        # Preparar DataFrame para os centros dos clusters
        cluster_centers = pd.DataFrame()

        # Definir limites geográficos do Brasil
        BR_BOUNDS = {
            "lat_min": -33.75,
            "lat_max": 5.27,
            "lon_min": -73.98,
            "lon_max": -34.79,
        }

        for cluster in range(self.n_clusters):
            # Obter dados do cluster
            cluster_cities = self.city_data[self.city_data["Cluster"] == cluster]
            cluster_data = self.df[self.df["Cluster"] == cluster]

            # Usar as coordenadas diretas do dataset, sem ponderação
            # Isso mantém a precisão geográfica original
            cluster_centers.loc[cluster, "Latitude"] = cluster_data["Latitude"].mean()
            cluster_centers.loc[cluster, "Longitude"] = cluster_data["Longitude"].mean()

            # Calcular estatísticas do cluster
            n_accidents = len(cluster_data)
            n_cities = len(cluster_cities)

            # Definir tamanho do marcador baseado no número de acidentes
            # Usando uma escala logarítmica ajustada para melhor visualização
            cluster_centers.loc[cluster, "marker_size"] = np.log1p(n_accidents) * 25

            # Preparar informações detalhadas para o tooltip
            char_info = [
                f"<b>Cluster {cluster}</b>",
                f"<b>Informações Gerais:</b>",
                f"• Total de Acidentes: {n_accidents}",
                f"• Número de Cidades: {n_cities}",
                f"• Coordenadas Centrais:",
                f"  - Latitude: {cluster_centers.loc[cluster, 'Latitude']:.4f}",
                f"  - Longitude: {cluster_centers.loc[cluster, 'Longitude']:.4f}",
                "",
                f"<b>Características Predominantes:</b>",
            ]

            # Adicionar características predominantes com formatação melhorada
            for feature in ["Causa_direta", "Natureza", "Periodo"]:
                if feature in cluster_data.columns and not cluster_data[feature].empty:
                    value = cluster_data[feature].mode().iloc[0]
                    pct = (cluster_data[feature] == value).mean() * 100
                    char_info.append(f"• {feature}: {value} ({pct:.1f}%)")

            # Juntar todas as informações com formatação HTML
            cluster_centers.loc[cluster, "characteristics"] = "<br>".join(char_info)

        # Criar o mapa com os novos parâmetros
        fig = px.scatter_mapbox(
            cluster_centers,
            lat="Latitude",
            lon="Longitude",
            size="marker_size",
            color=cluster_centers.index.astype(str),
            hover_data=["characteristics"],
            color_discrete_sequence=self.color_sequence,
            height=800,
            title="Distribuição Geográfica dos Clusters Categóricos",
        )

        # Configurar o layout do mapa para melhor visualização
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",  # White theme map
                center=dict(
                    # Centralizar no Brasil usando coordenadas médias
                    lat=-15.77,  # Latitude central do Brasil
                    lon=-47.92,  # Longitude central do Brasil
                ),
                zoom=3.5,
            ),
            showlegend=True,
            legend_title="Clusters",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(0, 0, 0, 0.8)",
            ),
            margin=dict(l=0, r=0, t=30, b=0),
        )

        # Ajustar os marcadores para melhor visibilidade
        fig.update_traces(
            marker=dict(
                sizeref=0.1,  # Controla a escala geral dos marcadores
                sizemin=15,  # Tamanho mínimo aumentado
                sizemode="area",
            ),
            hovertemplate="%{customdata[0]}<extra></extra>",
        )

        return fig

    def analyze_cluster_composition(self) -> pd.DataFrame:
        """
        Analyzes the composition of each cluster based on categorical features.
        """
        categorical_features = [
            "Causa_direta",
            "Causa_contibutiva",
            "Perimetro_Urbano",
            "Natureza",
            "Periodo",
        ]

        cluster_analysis = pd.DataFrame()

        for cluster in range(self.n_clusters):
            cluster_data = self.df[self.df["Cluster"] == cluster]

            # Calculate size and percentage of total
            size = len(cluster_data)
            percentage = (size / len(self.df)) * 100

            cluster_analysis.loc[cluster, "Tamanho"] = size
            cluster_analysis.loc[cluster, "Porcentagem do Total"] = percentage
            cluster_analysis.loc[cluster, "Número de Cidades"] = self.city_data[
                self.city_data["Cluster"] == cluster
            ]["Municipio"].nunique()

            # Analyze dominant categories
            for feature in categorical_features:
                if feature in cluster_data.columns:  # Check if feature exists
                    value_counts = cluster_data[feature].value_counts()
                    if not value_counts.empty:
                        most_common = value_counts.index[0]
                        percentage = (value_counts.iloc[0] / size) * 100

                        cluster_analysis.loc[cluster, f"{feature} Predominante"] = (
                            most_common
                        )
                        cluster_analysis.loc[cluster, f"{feature} %"] = percentage

        return cluster_analysis


class SparseKMeansVisualizer:
    """Handles visualization for Sparse K-means clustering results."""

    def __init__(
        self,
    ) -> None:
        self.logger = logging.getLogger(__name__)

    def create_sparse_cluster_map(
        self,
        df: pd.DataFrame,
        result: SparseKMeansResult,
        preprocess_data: ClusteringResult,
        n_clusters: int,
    ) -> go.Figure:
        """Creates an interactive map visualization of sparse clustering results."""

        try:
            # Add cluster assignments to DataFrame
            df_with_clusters = df.copy()
            df_with_clusters["cluster"] = result.labels

            # Calculate centroids
            centroids_df = pd.DataFrame()
            for cluster in range(n_clusters):
                mask = df_with_clusters["cluster"] == cluster
                weight = preprocess_data.original_values["num_acidentes"][mask]

                centroids_df.loc[cluster, "Latitude"] = np.average(
                    preprocess_data.original_values["Latitude"][mask], weights=weight
                )
                centroids_df.loc[cluster, "Longitude"] = np.average(
                    preprocess_data.original_values["Longitude"][mask], weights=weight
                )

            # Calculate cluster statistics
            cluster_stats = pd.DataFrame()

            for cluster in range(n_clusters):
                mask = df_with_clusters["cluster"] == cluster
                cluster_stats.loc[cluster, "total_acidentes"] = (
                    preprocess_data.original_values["num_acidentes"][mask].sum()
                )
                cluster_stats.loc[cluster, "total_interrupcao"] = (
                    preprocess_data.original_values["Interrupcao"][mask].sum()
                )
                cluster_stats.loc[cluster, "prejuizo_total"] = (
                    preprocess_data.original_values["Prejuizo_Financeiro"][mask].sum()
                )
                cluster_stats.loc[cluster, "principais_cidades"] = ", ".join(
                    df.loc[mask, "Municipio"].head(3)
                )

            # Create DataFrame for visualization
            centroids_df = centroids_df.join(cluster_stats)

            min_size, max_size = 20, 100
            if (
                centroids_df["total_acidentes"].max()
                != centroids_df["total_acidentes"].min()
            ):
                centroids_df["marker_size"] = (
                    centroids_df["total_acidentes"]
                    - centroids_df["total_acidentes"].min()
                ) / (
                    centroids_df["total_acidentes"].max()
                    - centroids_df["total_acidentes"].min()
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
                color_continuous_scale=px.colors.sequential.Redor[1:],
                labels={
                    "prejuizo_total": "Prejuízo Total (R$)",
                    "total_interrupcao": "Tempo Total de Interrupção (horas)",
                    "total_acidentes": "Total de Acidentes",
                    "principais_cidades": "Principais Cidades",
                },
            ).update_layout(
                mapbox_style="open-street-map",
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
                mapbox_style="open-street-map",
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

        except Exception as e:
            self.logger.error("Error creating sparse cluster map", exc_info=True)
            raise

    def _calculate_cluster_centroids(
        self, df: pd.DataFrame, result: SparseKMeansResult, lat_col: str, lon_col: str
    ) -> pd.DataFrame:
        """
        Calculates cluster centroids and relevant statistics.
        """
        self.logger.debug("Calculating cluster centroids and statistics")

        try:
            # Add cluster assignments to DataFrame
            df_with_clusters = df.copy()
            df_with_clusters["cluster"] = result.labels

            centroids_df = pd.DataFrame()
            n_clusters = len(np.unique(result.labels))

            for cluster in range(n_clusters):
                cluster_data = df_with_clusters[df_with_clusters["cluster"] == cluster]

                # Calculate weighted centroids based on accident counts
                weights = (
                    cluster_data["num_acidentes"]
                    if "num_acidentes" in cluster_data.columns
                    else None
                )

                centroids_df.loc[cluster, "Latitude"] = np.average(
                    cluster_data[lat_col], weights=weights
                )
                centroids_df.loc[cluster, "Longitude"] = np.average(
                    cluster_data[lon_col], weights=weights
                )

                # Calculate cluster statistics
                centroids_df.loc[cluster, "size"] = len(cluster_data)
                centroids_df.loc[cluster, "numero_cidades"] = cluster_data[
                    "Municipio"
                ].nunique()

                # Calculate total accidents and financial impact
                if "num_acidentes" in cluster_data.columns:
                    centroids_df.loc[cluster, "total_acidentes"] = cluster_data[
                        "num_acidentes"
                    ].sum()
                if "Prejuizo_Financeiro" in cluster_data.columns:
                    centroids_df.loc[cluster, "prejuizo_total"] = cluster_data[
                        "Prejuizo_Financeiro"
                    ].sum()

                # Get dominant features for this cluster
                self._add_cluster_characteristics(
                    centroids_df, cluster, result, cluster_data
                )

            return centroids_df

        except Exception as e:
            self.logger.error("Error calculating cluster centroids", exc_info=True)
            raise

    def _add_cluster_characteristics(
        self,
        centroids_df: pd.DataFrame,
        cluster: int,
        result: SparseKMeansResult,
        cluster_data: pd.DataFrame,
    ):
        """
        Adds characteristic features and their importance for each cluster.
        """
        try:
            # Get top features for this cluster based on the sparse weights
            cluster_center = result.centers[cluster]
            important_features = []

            for feat, weight in zip(result.selected_features, result.weights):
                if weight > 0:  # Only consider features selected by sparse k-means
                    feat_value = cluster_center[result.selected_features.index(feat)]
                    important_features.append((feat, feat_value, weight))

            # Sort by absolute feature value * weight to get most characteristic features
            important_features.sort(key=lambda x: abs(x[1] * x[2]), reverse=True)

            # Store top 3 characteristic features
            top_features = important_features[:3]
            feature_desc = []
            for feat, value, weight in top_features:
                direction = "high" if value > 0 else "low"
                feature_desc.append(f"{feat} ({direction}, importance: {weight:.2f})")

            centroids_df.loc[cluster, "characteristic_features"] = "<br>".join(
                feature_desc
            )

        except Exception as e:
            self.logger.warning(
                f"Error adding characteristics for cluster {cluster}", exc_info=True
            )

    def _create_map_figure(self, centroids_df: pd.DataFrame) -> go.Figure:
        """
        Creates the final map visualization with all components.
        """
        self.logger.debug("Creating map figure with cluster visualization")

        try:
            # Scale marker sizes based on number of accidents
            min_size, max_size = 20, 100
            if "total_acidentes" in centroids_df.columns:
                centroids_df["marker_size"] = self._scale_marker_sizes(
                    centroids_df["total_acidentes"], min_size, max_size
                )
            else:
                centroids_df["marker_size"] = min_size

            # Create the map
            map_figure = px.scatter_mapbox(
                centroids_df,
                lat="Latitude",
                lon="Longitude",
                size="marker_size",
                color=centroids_df.index.astype(str),
                custom_data=[
                    "size",
                    "numero_cidades",
                    "total_acidentes",
                    "prejuizo_total",
                    "characteristic_features",
                ],
                zoom=4,
                height=800,
                title="Distribuição Geográfica dos Clusters (Sparse K-means)",
            )

            # Update layout and styling
            map_figure.update_layout(
                mapbox_style="open-street-map",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )

            # Customize hover template
            map_figure.update_traces(
                hovertemplate=(
                    "<b>Cluster %{customdata[0]}</b><br>"
                    "Cidades: %{customdata[1]}<br>"
                    "Total de Acidentes: %{customdata[2]}<br>"
                    "Impacto Financeiro: R$ %{customdata[3]:,.2f}<br>"
                    "<b>Características Principais:</b><br>"
                    "%{customdata[4]}"
                    "<extra></extra>"
                )
            )

            return map_figure

        except Exception as e:
            self.logger.error("Error creating map figure", exc_info=True)
            raise

    @staticmethod
    def _scale_marker_sizes(
        values: pd.Series, min_size: float, max_size: float
    ) -> pd.Series:
        """
        Scales values to marker sizes using a logarithmic scale.
        """
        if values.max() == values.min():
            return pd.Series([min_size] * len(values))

        scaled = np.log1p(values - values.min())
        scaled = (scaled - scaled.min()) / (scaled.max() - scaled.min())
        return scaled * (max_size - min_size) + min_size

    def create_feature_importance_plot(self, result: SparseKMeansResult) -> go.Figure:
        """
        Creates an interactive bar plot of feature importance scores.
        """
        self.logger.debug("Creating feature importance visualization")

        try:
            # Sort features by importance
            sorted_features = sorted(
                result.feature_importance.items(), key=lambda x: x[1], reverse=True
            )

            features, importance = zip(*sorted_features)

            # Create bar plot
            fig = go.Figure(
                [
                    go.Bar(
                        x=features,
                        y=importance,
                        marker_color="lightblue",
                        hovertemplate=(
                            "<b>%{x}</b><br>"
                            "Importância: %{y:.3f}<br>"
                            "<extra></extra>"
                        ),
                    )
                ]
            )

            # Update layout
            fig.update_layout(
                title={
                    "text": "Importância das Features no Sparse K-means",
                    "y": 0.95,
                    "x": 0.5,
                    "xanchor": "center",
                    "yanchor": "top",
                },
                xaxis_title="Features",
                yaxis_title="Score de Importância",
                showlegend=False,
                xaxis_tickangle=45,
                height=600,
                margin=dict(t=100, b=100),
                template="plotly_white",
            )

            # Add threshold line for selected features
            if result.weights is not None and len(result.weights) > 0:
                threshold = np.mean(result.weights) / 2
                fig.add_hline(
                    y=threshold,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Limiar de Seleção",
                    annotation_position="bottom right",
                )

            return fig

        except Exception as e:
            self.logger.error("Error creating feature importance plot", exc_info=True)
            raise

    def create_cluster_characteristics_table(
        self, df: pd.DataFrame, result: SparseKMeansResult
    ) -> pd.DataFrame:
        """
        Creates a detailed summary table of cluster characteristics.
        """
        self.logger.debug("Creating cluster characteristics summary")

        try:
            summary = pd.DataFrame()
            df_with_clusters = df.copy()
            df_with_clusters["Cluster"] = result.labels

            for cluster in range(len(np.unique(result.labels))):
                cluster_data = df_with_clusters[df_with_clusters["Cluster"] == cluster]

                # Basic statistics
                summary.loc[f"Cluster {cluster}", "Número de Cidades"] = cluster_data[
                    "Municipio"
                ].nunique()
                summary.loc[f"Cluster {cluster}", "Total de Acidentes"] = len(
                    cluster_data
                )

                # Financial impact
                if "Prejuizo_Financeiro" in cluster_data.columns:
                    total_loss = cluster_data["Prejuizo_Financeiro"].sum()
                    summary.loc[f"Cluster {cluster}", "Prejuízo Total"] = (
                        f"R$ {total_loss:,.2f}"
                    )

                # Top features
                feature_desc = []
                for feat in result.selected_features[:5]:  # Top 5 features
                    if feat in cluster_data.columns:
                        value = cluster_data[feat].mode().iloc[0]
                        pct = (cluster_data[feat] == value).mean() * 100
                        feature_desc.append(f"{feat}: {value} ({pct:.1f}%)")

                summary.loc[f"Cluster {cluster}", "Características Principais"] = (
                    "\n".join(feature_desc)
                )

            return summary

        except Exception as e:
            self.logger.error(
                "Error creating cluster characteristics table", exc_info=True
            )
            raise
