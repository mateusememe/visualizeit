"""
Railway Accidents Visualization Application
----------------------------------------
A Streamlit application for visualizing and analyzing railway accidents data from ANTT (Brazilian National Land Transportation Agency).
The app provides interactive visualizations including clustering analysis, maps, and comparative charts.

Author: Mateus Mendonça Monteiro
Version: 1.0.0
"""

from sklearn.cluster import KMeans
import streamlit as st
import pandas as pd
from typing import Dict

from utils.analyser import TimePeriodAnalyzer
from utils.data import DataLoader, DataProcessor
from utils.filter import DataFilter, SidebarFilters
from utils.visualizer import ClusteringAnalyzer, Visualizer

# Page Configuration
st.set_page_config(
    page_title="Visualização de Acidentes Ferroviários - ANTT",
    layout="wide",
    initial_sidebar_state="expanded",
)


def tabs_handler(
    visualizer: Visualizer, df_filtered: pd.DataFrame, filters: Dict
) -> None:
    (
        kmeans_numerical,
        kmeans_categorical,
        accident_map_tab,
        time_analysis,
        concessionaria_analysis,
    ) = st.tabs(
        [
            "K-Means Numérico",
            "K-Means Categórico",
            "Mapa de Acidentes",
            "Análise por Período",
            "Análise por Concessionária",
        ]
    )

    with kmeans_numerical:
        st.header("Clusterização de Acidentes por Cidade (K-Means Numérico)")

        # Get numerical clustering results
        clustering_result = DataProcessor.preprocess_for_kmeans(df_filtered)

        col1, col2 = st.columns([3, 1])

        with col1:
            elbow_plot, optimal_clusters = DataProcessor.analyze_optimal_clusters(
                data=clustering_result.city_data, features=clustering_result.features
            )
            st.plotly_chart(elbow_plot, use_container_width=True)

        with col2:
            st.markdown(
                f"""
                ### Análise do Elbow Method

                Baseado na análise:
                - Número ótimo sugerido de clusters: **{optimal_clusters}**
                - Range recomendado: **{optimal_clusters-2}** a **{optimal_clusters+2}**
            """
            )

        # Create cluster map with current number of clusters
        cluster_map = visualizer.create_cluster_map(
            clustering_result, filters["n_clusters"]
        )
        st.plotly_chart(cluster_map)

        if st.checkbox("Mostrar dados brutos cluster KMeans Numérico"):
            st.write(clustering_result.city_data)

    with kmeans_categorical:
        st.header("Clusterização de Acidentes por Cidade (K-Means Categórico)")

        # Get categorical clustering results
        categorical_result = DataProcessor.preprocess_for_categorical_kmeans(
            df_filtered
        )

        # Perform clustering
        kmeans = KMeans(n_clusters=filters["n_clusters"], random_state=42)
        cluster_labels = kmeans.fit_predict(
            categorical_result.city_data[categorical_result.features]
        )

        # Initialize cluster analyzer with both original and city data
        cluster_analyzer = ClusteringAnalyzer(
            df=df_filtered,
            cluster_labels=cluster_labels,
            n_clusters=filters["n_clusters"],
            city_data=categorical_result.city_data,
        )

        col1, col2 = st.columns([3, 1])

        with col1:
            cat_elbow_plot, cat_optimal_clusters = (
                DataProcessor.analyze_optimal_clusters(
                    categorical_result.city_data,
                    categorical_result.features,
                    max_clusters=25,  # Smaller range for categorical due to more features
                )
            )
            st.plotly_chart(cat_elbow_plot, use_container_width=True)

        with col2:
            st.markdown(
                f"""
                ### Análise do Elbow Method (Categórico)

                Baseado na análise:
                - Número ótimo sugerido de clusters: **{cat_optimal_clusters}**
                - Range recomendado: **{cat_optimal_clusters-2}** a **{cat_optimal_clusters+2}**
            """
            )

        # Show geographical distribution
        st.subheader("Distribuição Geográfica dos Clusters")
        geo_map = cluster_analyzer.create_geographical_cluster_map()
        st.plotly_chart(geo_map, use_container_width=True)

        # Show cluster analysis
        st.subheader("Análise Detalhada dos Clusters")
        cluster_analysis = cluster_analyzer.analyze_cluster_composition()

        # Show detailed metrics
        if st.checkbox("Mostrar métricas detalhadas dos clusters"):
            st.dataframe(
                cluster_analysis.style.format(
                    {
                        "Tamanho": "{:.0f}",
                        "Porcentagem do Total": "{:.1f}%",
                        "Número de Cidades": "{:.0f}",
                        "Causa_direta %": "{:.1f}%",
                        "Causa_contribuitiva %": "{:.1f}%",
                        "Natureza %": "{:.1f}%",
                        "Periodo %": "{:.1f}%",
                    }
                )
            )

    with time_analysis:
        st.header("Análise Temporal de Acidentes")

        # Initialize time analyzer
        time_analyzer = TimePeriodAnalyzer(df_filtered)

        # Show hourly distribution
        hourly_dist = time_analyzer.create_hourly_distribution()
        st.plotly_chart(hourly_dist, use_container_width=True)

        # Show time patterns summary
        st.subheader("Resumo por Período")
        time_patterns = time_analyzer.analyze_time_patterns()
        st.dataframe(
            time_patterns.style.format(
                {"Número de Acidentes": "{:,.0f}", "Porcentagem do Total": "{:.1f}%"}
            )
        )

        # Show improved categorical analysis using heatmaps
        st.subheader("Distribuição de Categorias por Período")
        time_categories = time_analyzer.create_category_time_heatmaps()
        st.plotly_chart(time_categories, use_container_width=True)

        # Add explanatory text with insights
        st.markdown(
            """
        ### Interpretação dos Heatmaps

        Os mapas de calor acima mostram:
        - A intensidade da cor indica a concentração de ocorrências
        - Valores mais escuros representam maior concentração
        - Cada célula mostra a porcentagem exata
        - Padrões temporais específicos para cada categoria

        Esta visualização permite identificar facilmente:
        - Categorias predominantes em cada período
        - Períodos com maior variação de ocorrências
        - Padrões de concentração ao longo do dia
        """
        )

    with accident_map_tab:
        visualizer.show_accident_map(df_filtered)

    with concessionaria_analysis:
        visualizer.show_concessionaria_analysis(df_filtered)


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
