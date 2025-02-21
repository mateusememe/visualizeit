from dataclasses import dataclass
import numpy as np
import streamlit as st
from typing import Dict, List, NamedTuple, Tuple
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import plotly.graph_objects as go
import logging
from utils.analyser import ClusterAnalyzer
from utils.processor import PerimeterProcessor


@dataclass
class CategoricalClusteringResult:
    """Data class to store categorical clustering analysis results."""

    city_data: pd.DataFrame
    features: List[str]
    scaler: StandardScaler
    original_values: Dict[str, pd.Series]
    feature_names: List[str]  # Store transformed feature names


@dataclass
class ClusteringResult:
    """Data class to store clustering analysis results."""

    city_data: pd.DataFrame
    features: List[str]
    scaler: StandardScaler
    original_values: Dict[str, pd.Series]


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
            "datasets/updated/acidents_ferroviarios_2020_2024_com_coords.csv",
            encoding="UTF-8",
            sep=",",
        )
        df = df.rename(columns={
            'Interrupção': 'Interrupcao',
            'Prejuízo_Financeiro': 'Prejuizo_Financeiro',
            'Perímetro_Urbano': 'Perimetro_Urbano',
            'Quilômetro_Inicial': 'Quilometro_Inicial'
        })
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
        perimeter_processor = PerimeterProcessor()
        df = perimeter_processor.process_and_validate(df)
        return df


class DataProcessor:
    """Handles data processing and feature engineering."""

    def __init__(self):
        """Initialize DataProcessor with logger."""
        self.logger = logging.getLogger(__name__)

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

    @staticmethod
    def _assign_period(hour: pd.Timestamp) -> int:
        """
        Assigns a period number based on the hour of occurrence.

        Args:
            hour (pd.Timestamp): Hour of occurrence

        Returns:
            int: Period number (1-4)
        """
        hour_val = hour.hour
        if 0 <= hour_val < 6:
            return 1
        elif 6 <= hour_val < 12:
            return 2
        elif 12 <= hour_val < 18:
            return 3
        else:
            return 4

    @staticmethod
    def preprocess_for_categorical_kmeans(
        df: pd.DataFrame,
    ) -> CategoricalClusteringResult:
        """
        Preprocesses data for categorical K-means clustering analysis.

        Args:
            df (pd.DataFrame): Input DataFrame with railway accidents data

        Returns:
            CategoricalClusteringResult: Processed data ready for clustering analysis
        """
        # Filter date range (keeping consistent with numerical preprocessing)
        df = df[
            (df["Data_Ocorrencia"] >= "2020-12-01")
            & (df["Data_Ocorrencia"] <= "2024-07-31")
        ]

        # Create period feature
        df["Periodo"] = df["Hora_Ocorrencia"].apply(DataProcessor._assign_period)

        # Prepare categorical features for encoding
        categorical_features = [
            "Causa_direta",
            "Causa_contibutiva",
            "Perimetro_Urbano",
            "Natureza",
            "Periodo",
        ]

        # Initialize OneHotEncoder
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

        # Fit and transform categorical features
        encoded_features = encoder.fit_transform(df[categorical_features])

        # Get feature names after encoding
        feature_names = encoder.get_feature_names_out(categorical_features)

        # Add geographical features
        geographical_features = ["Latitude", "Longitude"]

        # Scale geographical features
        scaler = StandardScaler()
        scaled_geo = scaler.fit_transform(df[geographical_features])

        # Combine encoded categorical and scaled geographical features
        final_features = np.hstack([encoded_features, scaled_geo])

        # Create final feature names list
        all_feature_names = list(feature_names) + geographical_features

        # Aggregate data by city for clustering
        city_data = pd.DataFrame(
            final_features, columns=all_feature_names, index=df.index
        )

        # Add city information
        city_data["Municipio"] = df["Municipio"]

        # Group by city and aggregate
        city_data = city_data.groupby("Municipio").agg(
            {**{col: "mean" for col in all_feature_names}, "Municipio": "first"}
        )

        # Store original values for reference
        original_values = {
            "Latitude": df.groupby("Municipio")["Latitude"].first(),
            "Longitude": df.groupby("Municipio")["Longitude"].first(),
            "accident_count": df.groupby("Municipio").size(),
        }

        return CategoricalClusteringResult(
            city_data=city_data,
            features=all_feature_names,
            scaler=scaler,
            original_values=original_values,
            feature_names=all_feature_names,
        )

    @staticmethod
    def analyze_optimal_clusters(
        data: pd.DataFrame,
        features: List[str],
        max_clusters: int = 25,
        min_clusters: int = 5,
    ) -> Tuple[go.Figure, int]:
        """
        Analyzes and suggests optimal number of clusters.

        Args:
            data (pd.DataFrame): Input data
            features (List[str]): Features to use for clustering
            max_clusters (int): Maximum number of clusters to test
            min_clusters (int): Minimum number of clusters to test

        Returns:
            Tuple[go.Figure, int]: Elbow plot and suggested optimal number of clusters
        """
        # Calculate metrics
        inertias = ClusterAnalyzer.calculate_clustering_metrics(
            data[features].values, max_clusters, min_clusters
        )

        # Create visualization
        elbow_plot = ClusterAnalyzer.create_elbow_plot(inertias, min_clusters)

        # Find optimal number of clusters using the elbow method
        # Calculate the angle of the elbow
        angles = []
        for i in range(1, len(inertias) - 1):
            angle = np.rad2deg(
                np.arctan2(inertias[i - 1] - inertias[i], 1)
                - np.arctan2(inertias[i] - inertias[i + 1], 1)
            )
            angles.append(angle)

        # Find the point with the maximum angle change
        optimal_clusters = angles.index(max(angles)) + min_clusters + 1

        return elbow_plot, optimal_clusters
