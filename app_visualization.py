"""
Railway Accidents Visualization Application
----------------------------------------
A Streamlit application for visualizing and analyzing railway accidents data from ANTT (Brazilian National Land Transportation Agency).
The app provides interactive visualizations including clustering analysis, maps, and comparative charts.

Author: Mateus Mendonça Monteiro
Version: 1.0.0
"""

import logging
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler
import streamlit as st
import pandas as pd
from typing import Dict

from models.robust_sparse_kmeans import SparseKMeansProcessor
from utils.analyser import SparseClusterAnalyzer, TimePeriodAnalyzer
from utils.comparison import ClusteringComparison, analyze_clustering_differences, generate_detailed_report
from utils.data import DataLoader, DataProcessor
from utils.filter import DataFilter, SidebarFilters
from utils.logging import LoggerSetup
from utils.visualizer import ClusteringAnalyzer, SparseKMeansVisualizer, Visualizer

# Page Configuration
st.set_page_config(
    page_title="Visualização de Acidentes Ferroviários - ANTT",
    layout="wide",
    initial_sidebar_state="expanded",
)


class Main:
    def __init__(self):
        logger_setup = LoggerSetup()
        self.logger = logger_setup.setup_logger(
            "visualize.it", "visualize.it", logging.DEBUG
        )

    def tabs_handler(
        self, visualizer: Visualizer, df_filtered: pd.DataFrame, filters: Dict
    ) -> None:
        (
            kmeans_numerical,
            kmeans_categorical,
            robust_sparse_kmeans,
            clustering_comparison,
            accident_map_tab,
            time_analysis,
            concessionaria_analysis,
        ) = st.tabs(
            [
                "K-Means Numérico",
                "K-Means Categórico",
                "Robust Sparse K-Means",
                "Comparação de Métodos",
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
                    data=clustering_result.city_data,
                    features=clustering_result.features,
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

        with robust_sparse_kmeans:
            st.header("Análise de Clusters Esparsos (Sparse K-means)")

            try:
                # Process data for sparse clustering
                processor = SparseKMeansProcessor()
                sparse_visualizer = SparseKMeansVisualizer()
                analyzer = SparseClusterAnalyzer()

                # Get preprocessed data
                preprocess_result = DataProcessor.preprocess_for_kmeans(df_filtered)

                # Validate data before analysis
                if len(preprocess_result.city_data) < 2:
                    st.warning(
                        "Dados insuficientes para clustering. Ajuste os filtros."
                    )
                    st.stop()

                # Analyze optimal number of clusters
                col1, col2 = st.columns([3, 1])

                with col1:
                    try:
                        elbow_plot, optimal_clusters = (
                            analyzer.analyze_optimal_clusters(
                                preprocess_result.city_data[
                                    preprocess_result.features
                                ].values,
                                preprocess_result.features,
                                max_clusters=25,
                            )
                        )
                        st.plotly_chart(elbow_plot, use_container_width=True)
                    except Exception as elbow_error:
                        st.warning(
                            "Não foi possível realizar a análise do número ótimo de clusters. "
                            "Usando configuração padrão."
                            f"Erro: {elbow_error}"
                        )
                        optimal_clusters = min(5, optimal_clusters)
                        logging.error(f"Elbow analysis error: {str(elbow_error)}")

                with col2:
                    st.markdown(
                        f"""
                        ### Análise do Elbow Method

                        Baseado na análise:
                        - Número ótimo sugerido de clusters: **{optimal_clusters}**
                        - Range recomendado: **{optimal_clusters-2}** a **{optimal_clusters+2}**
                        """
                    )

                # Use minimum between user selection and data size
                n_clusters = min(
                    filters["n_clusters"], len(preprocess_result.city_data) - 1
                )

                # Perform clustering
                clustering_result = processor.perform_sparse_clustering(
                    city_data=preprocess_result.city_data,
                    features=preprocess_result.features,
                    n_clusters=n_clusters,
                )

                # Create visualizations
                if clustering_result:
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        cluster_map = sparse_visualizer.create_sparse_cluster_map(
                            preprocess_result.city_data,
                            clustering_result,
                            preprocess_result,
                            n_clusters,
                        )
                        st.plotly_chart(cluster_map, use_container_width=True)

                    with col2:
                        feature_importance = (
                            sparse_visualizer.create_feature_importance_plot(
                                clustering_result
                            )
                        )
                        st.plotly_chart(feature_importance, use_container_width=True)

                    # Show cluster characteristics
                    st.subheader("Características dos Clusters")
                    characteristics = (
                        sparse_visualizer.create_cluster_characteristics_table(
                            preprocess_result.city_data, clustering_result
                        )
                    )
                    st.dataframe(characteristics)
                else:
                    st.error(
                        "Não foi possível realizar o clustering com os parâmetros atuais."
                    )

            except Exception as e:
                st.error(f"Erro na análise de clusters esparsos: {str(e)}")
                logging.error("Error in sparse clustering analysis", exc_info=True)

        with clustering_comparison:
            st.header("Análise Comparativa: K-means vs Robust Sparse K-means")


            st.markdown(
                """
            ### Sobre a Análise Comparativa
            Esta análise compara o K-means tradicional com o Robust Sparse K-means,
            considerando múltiplos aspectos como estrutura dos clusters, importância
            das features e robustez a outliers.
            """
            )

            # Realizar análise comparativa
            try:

                # Verificar se estamos usando as colunas corretas
                required_features = [
                    "Data_Ocorrencia",
                    "Interrupcao",
                    "Prejuizo_Financeiro",
                ]

                # Criar dados para clustering
                clustering_data = pd.DataFrame()
                clustering_data = (df_filtered.groupby("Municipio")
                    .agg(
                        {
                            "Latitude": "first",
                            "Longitude": "first",
                            "Data_Ocorrencia": "count",
                            "Interrupcao": "sum",
                            "Prejuizo_Financeiro": "sum",
                        }
                    )
                .reset_index())
                clustering_data.rename(columns={"Data_Ocorrencia": "num_acidentes"}, inplace=True)
                
                # Contar acidentes por grupo (se necessário)
                # if "Data_Ocorrencia" in df_filtered.columns:
                #     accidents_count = df_filtered.groupby(df_filtered.index)[
                #         "Data_Ocorrencia"
                #     ].count()
                #     clustering_data["num_acidentes"] = accidents_count

                # Adicionar outras features
                # for feature in ["Interrupcao", "Prejuizo_Financeiro"]:
                #     if feature in df_filtered.columns:
                #         clustering_data[feature] = df_filtered[feature]

                # Verificar se temos todas as features necessárias
                if len(clustering_data.columns) < 3:
                    st.error(
                        """
                        Dados insuficientes para análise.
                        Necessitamos de informações sobre:
                        - Número de acidentes
                        - Tempo de interrupção
                        - Prejuízo financeiro
                    """
                    )
                    return

                # Preparar dados para clustering
                features = ["num_acidentes", "Interrupcao", "Prejuizo_Financeiro"]
                X = clustering_data[features].values
                feature_names = clustering_data[features].columns.tolist()
                # Normalizar dados
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # Criar e executar comparador
                comparator = ClusteringComparison(
                    X=X_scaled,
                    feature_names=feature_names,
                    n_clusters=filters["n_clusters"],
                    random_state=42,
                )

                # Análise principal
                metrics = comparator.fit_and_compare()

                # Criar e mostrar visualizações
                visualizations = comparator.create_comparison_visualizations()

                st.subheader("Comparação de Importância das Features")
                st.plotly_chart(
                    visualizations["feature_importance"], use_container_width=True
                )

                # st.subheader("Distribuição dos Clusters")
                # st.plotly_chart(
                #     visualizations["cluster_distribution"], use_container_width=True
                # )

                # # Análises adicionais conforme seleção do usuário
                # if show_stability:
                #     with st.spinner("Realizando análise de estabilidade..."):
                #         stability_metrics = comparator.analyze_cluster_stability(
                #             n_runs=n_runs
                #         )
                #         stability_viz = comparator.create_stability_visualization(
                #             stability_metrics
                #         )

                #         st.subheader("Análise de Estabilidade")
                #         st.plotly_chart(stability_viz, use_container_width=True)

                # if show_interpretability:
                #     st.subheader("Interpretabilidade dos Clusters")
                #     interpretability = comparator.compare_cluster_interpretability()
                #     st.dataframe(
                #         interpretability.style.background_gradient(
                #             subset=["Valor Médio"], cmap="RdYlBu"
                #         )
                #     )
                st.subheader("Análise SHAP dos Modelos")

                with st.spinner("Realizando análise SHAP..."):
                    analysis_results, comparison_table = analyze_clustering_differences(
                        kmeans_model=comparator.kmeans,
                        rskc_model=comparator.rskc,
                        X=X_scaled,
                        feature_names=feature_names,
                    )

                    # Mostrar visualizações SHAP
                    st.plotly_chart(
                        analysis_results["visualizations"]["feature_importance"],
                        use_container_width=True,
                    )

                    # Mostrar tabela comparativa
                    st.subheader("Comparação Detalhada das Features")
                    st.dataframe(
                        comparison_table.style.background_gradient(
                            subset=["Diferença Relativa"], cmap="RdYlBu"
                        )
                    )

                    # Mostrar análise de interações
                    st.subheader("Análise de Interações entre Features")
                    st.dataframe(
                        analysis_results[
                            "interaction_analysis"
                        ].style.background_gradient(cmap="viridis")
                    )
                # # Gerar e mostrar relatório
                # st.subheader("Relatório Detalhado")
                # report = generate_detailed_report(
                #     comparator, stability_metrics if show_stability else None
                # )
                # st.markdown(report)

            except Exception as e:
                st.error(
                    f"""
                    Erro na análise comparativa: {str(e)}

                    Detalhes do erro:
                    - Tipo de erro: {type(e).__name__}
                    - Mensagem: {str(e)}

                    Sugestões de resolução:
                    1. Verifique se todas as colunas necessárias estão presentes
                    2. Confira se os dados estão no formato correto
                    3. Ajuste os parâmetros de clustering se necessário
                """
                )
                st.exception(e)
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
                    {
                        "Número de Acidentes": "{:,.0f}",
                        "Porcentagem do Total": "{:.1f}%",
                    }
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

    def execute(self):
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
        self.tabs_handler(visualizer, df_filtered, filters)


if __name__ == "__main__":
    main = Main()
    main.execute()
