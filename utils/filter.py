import streamlit as st
from typing import Any, Dict

import pandas as pd


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
