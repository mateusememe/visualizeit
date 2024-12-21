import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
from typing import Tuple, List
import numpy as np
from plotly.subplots import make_subplots
import logging

from models.robust_sparse_kmeans import RobustSparseKMeans


class ClusterAnalyzer:
    """Handles cluster analysis and optimization."""

    def __init__(self):
        """Initialize ClusterAnalyzer with logger."""
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def calculate_clustering_metrics(
        data: np.ndarray,
        max_clusters: int = 25,
        min_clusters: int = 2,
        random_state: int = 42,
    ) -> List[float]:
        """
        Calculates clustering metrics (inertia and silhouette score) for different numbers of clusters.

        Args:
            data (np.ndarray): Input data for clustering
            max_clusters (int): Maximum number of clusters to test
            min_clusters (int): Minimum number of clusters to test
            random_state (int): Random state for reproducibility

        Returns:
            List[float]: List containing inertia
        """
        inertias = []

        n_clusters_range = range(min_clusters, max_clusters + 1)

        for n_clusters in n_clusters_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            kmeans.fit(data)
            inertias.append(kmeans.inertia_)

        return inertias

    @staticmethod
    def create_elbow_plot(
        inertias: List[float],
        min_clusters: int = 2,
        title: str = "Análise do Número Ideal de Clusters",
    ) -> go.Figure:
        """
        Creates an interactive plot showing inertia scores.

        Args:
            inertias (List[float]): List of inertia values

            min_clusters (int): Minimum number of clusters tested
            title (str): Plot title

        Returns:
            go.Figure: Plotly figure containing the elbow plot
        """
        n_clusters_range = list(range(min_clusters, min_clusters + len(inertias)))

        # Create figure with secondary y-axis
        fig = go.Figure()

        # Add inertia trace
        fig.add_trace(
            go.Scatter(
                x=n_clusters_range,
                y=inertias,
                name="Inertia",
                mode="lines+markers",
                line=dict(color="blue"),
                yaxis="y",
            )
        )

        # Update layout with second y-axis
        fig.update_layout(
            title=title,
            xaxis=dict(title="Número de Clusters", gridcolor="lightgray"),
            yaxis=dict(title="Inertia", gridcolor="lightgray"),
            hovermode="x unified",
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
            plot_bgcolor="white",
        )

        return fig


class TimePeriodAnalyzer:
    """
    Analyzes and visualizes accident patterns based on time periods throughout the day.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the analyzer with dataset and period definitions.

        Args:
            df (pd.DataFrame): Input DataFrame with Hora_Ocorrencia in HH:MM format
        """
        self.df = df.copy()
        self.logger = logging.getLogger(__name__)

        # Define our time periods with their labels
        self.period_labels = {
            1: "00:00-06:00",
            2: "06:00-12:00",
            3: "12:00-18:00",
            4: "18:00-00:00",
        }
        # Process the time data
        self._process_time_periods()

    def _time_to_period(self, time_str: str) -> int:
        """
        Convert a time string to its corresponding period number.

        Args:
            time_str (str): Time in HH:MM format

        Returns:
            int: Period number (1-4)
        """
        try:
            # Parse the time string
            hour = int(time_str.split(":")[0])

            # Determine period
            if 0 <= hour < 6:
                return 1
            elif 6 <= hour < 12:
                return 2
            elif 12 <= hour < 18:
                return 3
            else:
                return 4
        except (ValueError, AttributeError, IndexError):
            return None

    def _process_time_periods(self):
        """
        Process time data and add period information to the DataFrame.
        """
        # Convert Hora_Ocorrencia to period numbers
        self.df["Periodo"] = (
            self.df["Hora_Ocorrencia"].astype(str).apply(self._time_to_period)
        )

        # Add hour for detailed time analysis
        self.df["Hora"] = (
            self.df["Hora_Ocorrencia"]
            .astype(str)
            .apply(lambda x: int(x.split(":")[0]) if ":" in str(x) else None)
        )

    def create_hourly_distribution(self) -> go.Figure:
        """
        Creates a detailed hourly distribution visualization of accidents.
        """
        hourly_counts = self.df["Hora"].value_counts().sort_index()

        fig = go.Figure()

        # Add hour-by-hour distribution
        fig.add_trace(
            go.Bar(
                x=hourly_counts.index,
                y=hourly_counts.values,
                name="Acidentes por Hora",
                marker_color="lightblue",
                hovertemplate=(
                    "Hora: %{x}:00<br>"
                    + "Número de Acidentes: %{y}<br>"
                    + "<extra></extra>"
                ),
            )
        )

        # Add period separators and annotations
        for period_start in [0, 6, 12, 18]:
            fig.add_vline(
                x=period_start, line_dash="dash", line_color="gray", opacity=0.5
            )

        # Add period labels
        for period, label in self.period_labels.items():
            start_hour = [0, 6, 12, 18][period - 1]
            fig.add_annotation(
                x=start_hour + 2,
                y=max(hourly_counts) * 1.1,
                text=label,
                showarrow=False,
                font=dict(size=10),
            )

        fig.update_layout(
            title="Distribuição de Acidentes por Hora do Dia",
            xaxis_title="Hora do Dia",
            yaxis_title="Número de Acidentes",
            template="plotly_white",
            xaxis=dict(
                tickmode="array",
                ticktext=[f"{i:02d}:00" for i in range(24)],
                tickvals=list(range(24)),
            ),
            showlegend=False,
        )

        return fig

    def create_multi_category_time_analysis(self) -> go.Figure:
        """
        Creates a comprehensive visualization of categorical features by time period.
        """
        categorical_features = [
            "Causa_direta",
            "Causa_contibutiva",
            "Perimetro_Urbano",
            "Natureza",
        ]

        fig = make_subplots(
            rows=len(categorical_features),
            cols=1,
            subplot_titles=[
                f"Distribuição de {feat} por Período do Dia"
                for feat in categorical_features
            ],
            vertical_spacing=0.15,
        )

        colors = [
            "rgb(31, 119, 180)",
            "rgb(255, 127, 14)",
            "rgb(44, 160, 44)",
            "rgb(214, 39, 40)",
        ]

        for idx, feature in enumerate(categorical_features, 1):
            # Calculate distribution for each period
            feature_period_dist = (
                pd.crosstab(self.df[feature], self.df["Periodo"], normalize="columns")
                * 100
            )

            # Add traces for each period
            for period in sorted(feature_period_dist.columns):
                fig.add_trace(
                    go.Bar(
                        name=f"Período {self.period_labels[period]}",
                        x=feature_period_dist.index,
                        y=feature_period_dist[period],
                        text=np.round(feature_period_dist[period], 1),
                        textposition="auto",
                        marker_color=colors[period - 1],
                        showlegend=idx == 1,
                        hovertemplate=(
                            f"<b>{feature}</b><br>"
                            + "Categoria: %{x}<br>"
                            + "Porcentagem: %{y:.1f}%<br>"
                            + f"Período: {self.period_labels[period]}<br>"
                            + "<extra></extra>"
                        ),
                    ),
                    row=idx,
                    col=1,
                )

            # Update axes
            fig.update_yaxes(
                title_text="Porcentagem (%)", range=[0, 100], row=idx, col=1
            )

            fig.update_xaxes(
                title_text=feature,
                tickangle=45 if len(feature_period_dist.index) > 5 else 0,
                row=idx,
                col=1,
            )

        fig.update_layout(
            height=1200,
            title={
                "text": "Análise de Categorias por Período do Dia",
                "y": 0.98,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
            },
            barmode="group",
            showlegend=True,
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "right",
                "x": 1,
            },
            template="plotly_white",
            margin=dict(t=100, b=50, l=50, r=50),
        )

        return fig

    def analyze_time_patterns(self) -> pd.DataFrame:
        """
        Creates a summary of time-related patterns in the data.
        """
        summary = pd.DataFrame()

        # Calculate period statistics
        period_counts = self.df["Periodo"].value_counts()
        total_accidents = len(self.df)

        for period, label in self.period_labels.items():
            count = period_counts.get(period, 0)
            summary.loc[label, "Número de Acidentes"] = count
            summary.loc[label, "Porcentagem do Total"] = (count / total_accidents) * 100

            # Most common characteristics for each period
            period_data = self.df[self.df["Periodo"] == period]
            if not period_data.empty:
                summary.loc[label, "Causa Mais Comum"] = (
                    period_data["Causa_direta"].mode().iloc[0]
                    if not period_data["Causa_direta"].mode().empty
                    else "N/A"
                )
                summary.loc[label, "Natureza Mais Comum"] = (
                    period_data["Natureza"].mode().iloc[0]
                    if not period_data["Natureza"].mode().empty
                    else "N/A"
                )

        return summary

    def create_category_time_heatmaps(self) -> go.Figure:
        """
        Creates an improved visualization using heatmaps to show the relationship
        between categories and time periods.
        """
        categorical_features = [
            "Causa_direta",
            # "Causa_contibutiva",
            "Perimetro_Urbano",
            "Natureza",
        ]

        # Create subplots - one for each categorical feature
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                f"Distribuição de {feat} por Período do Dia"
                for feat in categorical_features
            ],
            vertical_spacing=0.16,
            horizontal_spacing=0.12,
        )

        # Define positions for subplots
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

        for feature, position in zip(categorical_features, positions):
            # Calculate percentage distribution
            crosstab = (
                pd.crosstab(self.df[feature], self.df["Periodo"], normalize="columns")
                * 100
            )

            # Reorder columns to match time sequence
            crosstab = crosstab.reindex(columns=sorted(crosstab.columns))

            # Create heatmap trace
            heatmap = go.Heatmap(
                z=crosstab.values,
                x=[self.period_labels[i] for i in crosstab.columns],
                y=crosstab.index,
                colorscale="YlOrRd",  # Yellow to Orange to Red colorscale
                text=np.round(crosstab.values, 1),
                texttemplate="%{text}%",
                textfont={"size": 10},
                hoverongaps=False,
                hovertemplate=(
                    "Período: %{x}<br>"
                    # + f"{feature}: %{y}<br>"
                    + "Porcentagem: %{z:.1f}%<br>"
                    + "<extra></extra>"
                ),
            )

            # Add trace to subplot
            fig.add_trace(heatmap, row=position[0], col=position[1])

            # Update axes labels
            fig.update_xaxes(
                title_text="Período do Dia",
                tickangle=45,
                row=position[0],
                col=position[1],
            )
            fig.update_yaxes(title_text=feature, row=position[0], col=position[1])

        # Update layout
        fig.update_layout(
            height=1000,
            title={
                "text": "Análise de Categorias por Período do Dia",
                "y": 0.95,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
            },
            template="plotly_white",
            showlegend=False,
        )

        # Add a single colorbar for all heatmaps
        fig.update_layout(
            coloraxis=dict(
                colorscale="YlOrRd",
                colorbar=dict(
                    title="Porcentagem (%)", titleside="right", x=1.02, y=0.5
                ),
            )
        )

        return fig


class SparseClusterAnalyzer:
    """
    Analyzes optimal number of clusters for Robust Sparse K-means.
    Includes elbow method analysis with feature importance consideration.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze_optimal_clusters(
        self,
        X: np.ndarray,
        feature_names: List[str],
        max_clusters: int = 25,
        min_clusters: int = 5,
        lasso_param: float = 0.1,
    ) -> Tuple[go.Figure, int]:
        """
        Analyzes optimal number of clusters with enhanced error checking.
        """

        try:
            # Initialize metrics storage
            inertias = []

            # Calculate metrics for each cluster size
            for n_clusters in range(min_clusters, max_clusters + 1):
                try:
                    model = RobustSparseKMeans(
                        n_clusters=n_clusters, lasso_param=lasso_param, random_state=42
                    )
                    model.fit(X, feature_names)
                    inertias.append(model.inertia_)

                except Exception as cluster_error:
                    self.logger.warning(
                        f"Error testing {n_clusters} clusters: {str(cluster_error)}"
                    )
                    continue

            # Create visualization
            sparse_elbow_plot = self._create_elbow_plot(
                inertias, min_clusters, feature_names
            )

            # Find optimal number of clusters
            optimal_clusters = self._find_elbow_point(inertias, min_clusters)

            return sparse_elbow_plot, optimal_clusters

        except Exception as e:
            self.logger.error(
                f"Error in optimal cluster analysis: {str(e)}", exc_info=True
            )
            raise

    def _create_elbow_plot(
        self,
        inertias: List[float],
        min_clusters: int,
        feature_names: List[str],
    ) -> go.Figure:
        """
        Creates elbow plot with enhanced error handling and data validation.
        """
        try:
            # Validate inputs
            if not inertias:
                raise ValueError("Empty metrics arrays")

            n_clusters_range = list(range(min_clusters, min_clusters + len(inertias)))
            total_features = len(feature_names)

            # Create figure
            fig = go.Figure()

            # Add inertia trace with error handling
            fig.add_trace(
                go.Scatter(
                    x=n_clusters_range,
                    y=inertias,
                    name="Inertia",
                    mode="lines+markers",
                    line=dict(color="blue"),
                    yaxis="y",
                )
            )

            # Update layout
            fig.update_layout(
                title="Análise do Número Ideal de Clusters (Sparse K-means)",
                xaxis_title="Número de Clusters",
                yaxis_title="Inertia",
                yaxis2=dict(
                    title="Número de Features",
                    overlaying="y",
                    side="right",
                    range=[0, total_features],
                ),
                showlegend=True,
                plot_bgcolor="white",
            )

            return fig

        except Exception as e:
            self.logger.error(f"Error creating elbow plot: {str(e)}", exc_info=True)
            raise

    def _find_elbow_point(self, inertias: List[float], min_clusters: int) -> int:
        """
        Finds the optimal number of clusters using both inertia and feature selection.

        Args:
            inertias: List of inertia values
            feature_counts: List of selected feature counts
            min_clusters: Minimum number of clusters tested

        Returns:
            Optimal number of clusters
        """
        # Calculate inertia angles
        inertia_angles = []
        for i in range(1, len(inertias) - 1):
            angle = np.rad2deg(
                np.arctan2(inertias[i - 1] - inertias[i], 1)
                - np.arctan2(inertias[i] - inertias[i + 1], 1)
            )
            inertia_angles.append(angle)

        optimal_clusters = inertia_angles.index(max(inertia_angles)) + min_clusters + 1

        return optimal_clusters
