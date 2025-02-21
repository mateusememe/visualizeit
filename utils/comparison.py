from turtle import st
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from models.robust_sparse_kmeans import RobustSparseKMeans
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class KMeansFeatureImportance:
    """
    Calcula e analisa a importância das features para clustering K-means.

    Esta classe implementa múltiplas abordagens para entender como cada feature
    contribui para a formação dos clusters, incluindo:
    1. Análise de variância entre/dentro dos clusters
    2. Contribuição para a separabilidade dos clusters
    3. Correlação com a estrutura dos clusters
    """

    def __init__(self, kmeans_model, X: np.ndarray, feature_names: List[str]):
        """
        Inicializa o analisador de importância de features.

        Args:
            kmeans_model: Modelo K-means já treinado
            X: Dados de entrada usados no treinamento
            feature_names: Nomes das features para interpretação
        """
        self.kmeans = kmeans_model
        self.X = X
        self.feature_names = feature_names
        self.n_features = X.shape[1]
        self.labels = kmeans_model.labels_
        self.centers = kmeans_model.cluster_centers_

    def calculate_feature_importance(self) -> Dict[str, float]:
        """
        Calcula a importância das features usando múltiplas métricas.

        O método combina diferentes abordagens para criar um score robusto:
        - Razão de variância entre/dentro dos clusters
        - Contribuição para a inércia total
        - Capacidade de discriminação entre clusters

        Returns:
            Dicionário com scores de importância para cada feature
        """
        # Calcular variância entre clusters (between-cluster variance)
        between_var = np.var(self.centers, axis=0)

        # Calcular variância dentro dos clusters (within-cluster variance)
        within_var = np.zeros(self.n_features)
        for k in range(len(self.centers)):
            mask = self.labels == k
            if np.any(mask):
                within_var += np.var(self.X[mask], axis=0)

        # Evitar divisão por zero
        within_var = np.where(within_var == 0, np.inf, within_var)

        # Calcular F-statistic como medida de importância
        f_statistic = between_var / within_var

        # Calcular contribuição para inércia
        inertia_contribution = np.zeros(self.n_features)
        for k in range(len(self.centers)):
            mask = self.labels == k
            if np.any(mask):
                diff = self.X[mask] - self.centers[k]
                inertia_contribution += np.sum(diff**2, axis=0)

        # Normalizar contribuições
        f_statistic_norm = f_statistic / np.sum(f_statistic)
        inertia_norm = 1 - (inertia_contribution / np.sum(inertia_contribution))

        # Combinar métricas (média ponderada)
        importance_scores = 0.7 * f_statistic_norm + 0.3 * inertia_norm

        # Criar dicionário de importância
        feature_importance = {
            name: score for name, score in zip(self.feature_names, importance_scores)
        }

        return feature_importance

    def analyze_cluster_separation(self) -> pd.DataFrame:
        """
        Analisa como cada feature contribui para a separação entre clusters.

        Returns:
            DataFrame com métricas detalhadas de separação por feature
        """
        analysis = pd.DataFrame(index=self.feature_names)

        for i, feature in enumerate(self.feature_names):
            # Calcular distância média entre centroides para esta feature
            center_distances = np.zeros((len(self.centers), len(self.centers)))
            for k1 in range(len(self.centers)):
                for k2 in range(k1 + 1, len(self.centers)):
                    dist = abs(self.centers[k1, i] - self.centers[k2, i])
                    center_distances[k1, k2] = center_distances[k2, k1] = dist

            # Calcular métricas de separação
            analysis.loc[feature, "Distância Média Entre Clusters"] = np.mean(
                center_distances[center_distances > 0]
            )

            # Calcular sobreposição entre clusters
            overlap_count = 0
            total_comparisons = 0

            for k1 in range(len(self.centers)):
                mask1 = self.labels == k1
                if not np.any(mask1):
                    continue

                values1 = self.X[mask1, i]
                min1, max1 = np.min(values1), np.max(values1)

                for k2 in range(k1 + 1, len(self.centers)):
                    mask2 = self.labels == k2
                    if not np.any(mask2):
                        continue

                    values2 = self.X[mask2, i]
                    min2, max2 = np.min(values2), np.max(values2)

                    # Verificar sobreposição
                    if min1 <= max2 and min2 <= max1:
                        overlap_count += 1
                    total_comparisons += 1

            if total_comparisons > 0:
                analysis.loc[feature, "Taxa de Sobreposição"] = (
                    overlap_count / total_comparisons
                )
            else:
                analysis.loc[feature, "Taxa de Sobreposição"] = np.nan

        return analysis


def create_feature_importance_visualization(
    feature_importance: Dict[str, float], separation_analysis: pd.DataFrame
) -> go.Figure:
    """
    Cria uma visualização interativa da importância das features.
    """
    # Ordenar features por importância
    sorted_features = sorted(
        feature_importance.items(), key=lambda x: x[1], reverse=True
    )
    features, importance = zip(*sorted_features)

    # Criar figura com subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            "Importância das Features no K-means",
            "Análise de Separação dos Clusters",
        ),
        vertical_spacing=0.3,
        heights=[0.6, 0.4],
    )

    # Adicionar barras de importância
    fig.add_trace(
        go.Bar(
            x=features,
            y=importance,
            marker_color="rgb(55, 83, 109)",
            name="Importância Global",
            hovertemplate=(
                "<b>%{x}</b><br>" + "Importância: %{y:.3f}<br>" + "<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    # Adicionar métricas de separação
    fig.add_trace(
        go.Scatter(
            x=features,
            y=separation_analysis["Distância Média Entre Clusters"],
            mode="lines+markers",
            name="Distância Entre Clusters",
            line=dict(color="rgb(26, 118, 255)"),
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=features,
            y=separation_analysis["Taxa de Sobreposição"],
            mode="lines+markers",
            name="Taxa de Sobreposição",
            line=dict(color="rgb(214, 39, 40)"),
        ),
        row=2,
        col=1,
    )

    # Atualizar layout
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="Análise de Importância das Features no K-means",
        template="plotly_white",
    )

    # Atualizar eixos
    fig.update_xaxes(tickangle=45)

    return fig


class ClusteringComparison:
    """
    Realiza uma análise comparativa detalhada entre K-means tradicional e RSKC.

    Esta classe implementa métricas e visualizações para entender como os dois
    métodos diferem em termos de:
    - Estrutura dos clusters formados
    - Seleção e importância das features
    - Robustez a outliers
    - Interpretabilidade dos resultados
    """

    def __init__(
        self,
        X: np.ndarray,
        feature_names: List[str],
        n_clusters: int = 5,
        random_state: int = 42,
    ):
        """
        Inicializa a análise comparativa.

        Args:
            X: Dados de entrada para clustering
            feature_names: Nomes das features para interpretação
            n_clusters: Número de clusters a serem formados
            random_state: Semente aleatória para reprodutibilidade
        """
        self.X = X
        self.feature_names = feature_names
        self.n_clusters = n_clusters
        self.random_state = random_state

        # Inicializar modelos
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)

        self.rskc = RobustSparseKMeans(
            n_clusters=n_clusters,
            # alpha=0.1,  # Parâmetro de trimming para RSKC
            max_iter=100,
            random_state=random_state,
        )

        # Resultados serão armazenados aqui
        self.kmeans_result = None
        self.rskc_result = None
        self.comparison_metrics = None

    def fit_and_compare(self) -> Dict:
        """
        Ajusta ambos os modelos e computa métricas comparativas.

        Returns:
            Dicionário com métricas comparativas detalhadas
        """
        # Ajustar K-means
        self.kmeans.fit(self.X)
        kmeans_analyzer = KMeansFeatureImportance(
            self.kmeans, self.X, self.feature_names
        )
        kmeans_importance = kmeans_analyzer.calculate_feature_importance()

        # Ajustar RSKC
        self.rskc.fit(self.X, self.feature_names)

        # Computar métricas comparativas
        metrics = {
            "kmeans": {
                "inertia": self.kmeans.inertia_,
                "feature_importance": kmeans_importance,
                "n_iterations": self.kmeans.n_iter_,
                "labels": self.kmeans.labels_,
            },
            "rskc": {
                "inertia": self.rskc.inertia_,
                "feature_importance": dict(zip(self.feature_names, self.rskc.weights_)),
                "n_iterations": self.rskc.n_iter_,
                "labels": self.rskc.labels_,
                "n_outliers": len(self.rskc.outliers_),
            },
        }

        # Calcular concordância entre os clusters
        metrics["cluster_agreement"] = self._calculate_cluster_agreement()

        self.comparison_metrics = metrics
        return metrics

    def _calculate_cluster_agreement(self) -> float:
        """
        Calcula o nível de concordância entre os clusters dos dois métodos.

        Returns:
            Score de concordância entre 0 e 1
        """
        from sklearn.metrics import adjusted_rand_score

        return adjusted_rand_score(self.kmeans.labels_, self.rskc.labels_)

    def create_comparison_visualizations(self) -> Dict[str, go.Figure]:
        """
        Cria um conjunto de visualizações comparativas.

        Returns:
            Dicionário com diferentes visualizações comparativas
        """
        if self.comparison_metrics is None:
            self.fit_and_compare()

        visualizations = {}

        # 1. Comparação de importância das features
        fig_importance = self._create_feature_importance_comparison()
        visualizations["feature_importance"] = fig_importance

        # 2. Distribuição dos clusters
        fig_clusters = self._create_cluster_distribution_comparison()
        visualizations["cluster_distribution"] = fig_clusters

        # 3. Análise de outliers (específico para RSKC)
        fig_outliers = self._create_outlier_analysis()
        visualizations["outlier_analysis"] = fig_outliers

        return visualizations

    def _create_feature_importance_comparison(self) -> go.Figure:
        """
        Cria visualização comparativa da importância das features.
        """
        kmeans_importance = self.comparison_metrics["kmeans"]["feature_importance"]
        rskc_importance = self.comparison_metrics["rskc"]["feature_importance"]

        fig = go.Figure()

        # Adicionar barras para K-means
        fig.add_trace(
            go.Bar(
                name="K-means",
                x=list(kmeans_importance.keys()),
                y=list(kmeans_importance.values()),
                marker_color="rgb(55, 83, 109)",
            )
        )

        # Adicionar barras para RSKC
        fig.add_trace(
            go.Bar(
                name="RSKC",
                x=list(rskc_importance.keys()),
                y=list(rskc_importance.values()),
                marker_color="rgb(26, 118, 255)",
            )
        )

        fig.update_layout(
            title="Comparação de Importância das Features",
            xaxis_title="Features",
            yaxis_title="Importância",
            barmode="group",
            template="plotly_white",
        )

        return fig

    def _create_cluster_distribution_comparison(self) -> go.Figure:
        """
        Cria visualização comparativa da distribuição dos clusters.
        """
        fig = make_subplots(
            rows=1, cols=2, subplot_titles=["K-means", "RSKC (com outliers)"]
        )

        # Distribuição K-means
        kmeans_dist = pd.Series(self.kmeans.labels_).value_counts()
        fig.add_trace(
            go.Bar(
                x=[f"Cluster {i}" for i in range(self.n_clusters)],
                y=[kmeans_dist.get(i, 0) for i in range(self.n_clusters)],
                name="K-means",
                marker_color="rgb(55, 83, 109)",
            ),
            row=1,
            col=1,
        )

        # Distribuição RSKC
        rskc_dist = pd.Series(self.rskc.labels_).value_counts()
        fig.add_trace(
            go.Bar(
                x=[f"Cluster {i}" for i in range(self.n_clusters)],
                y=[rskc_dist.get(i, 0) for i in range(self.n_clusters)],
                name="RSKC",
                marker_color="rgb(26, 118, 255)",
            ),
            row=1,
            col=2,
        )

        # Adicionar marca para outliers no RSKC
        if len(self.rskc.outliers_) > 0:
            fig.add_trace(
                go.Bar(
                    x=["Outliers"],
                    y=[len(self.rskc.outliers_)],
                    name="Outliers",
                    marker_color="red",
                ),
                row=1,
                col=2,
            )

        fig.update_layout(
            title="Distribuição dos Clusters", showlegend=True, template="plotly_white"
        )

        return fig

    def _create_outlier_analysis(self) -> go.Figure:
        """
        Cria visualização específica para análise de outliers do RSKC.
        """
        fig = go.Figure()

        # Separar pontos normais e outliers
        outlier_mask = np.isin(np.arange(len(self.X)), self.rskc.outliers_)
        normal_points = self.X[~outlier_mask]
        outlier_points = self.X[outlier_mask]

        # Calcular componentes principais para visualização 2D
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X)

        # Plotar pontos normais
        fig.add_trace(
            go.Scatter(
                x=X_pca[~outlier_mask, 0],
                y=X_pca[~outlier_mask, 1],
                mode="markers",
                name="Pontos Normais",
                marker=dict(
                    color=self.rskc.labels_[~outlier_mask],
                    colorscale="Viridis",
                    showscale=True,
                ),
            )
        )

        # Plotar outliers
        if len(self.rskc.outliers_) > 0:
            fig.add_trace(
                go.Scatter(
                    x=X_pca[outlier_mask, 0],
                    y=X_pca[outlier_mask, 1],
                    mode="markers",
                    name="Outliers",
                    marker=dict(color="red", symbol="x", size=10),
                )
            )

        fig.update_layout(
            title="Visualização de Outliers (PCA)",
            xaxis_title="Primeira Componente Principal",
            yaxis_title="Segunda Componente Principal",
            template="plotly_white",
        )

        return fig

    def analyze_cluster_stability(self, n_runs: int = 10) -> Dict:
        """
        Analisa a estabilidade dos clusters em múltiplas execuções.

        Esta função executa os algoritmos várias vezes para avaliar
        a consistência dos resultados.

        Args:
            n_runs: Número de execuções para testar estabilidade

        Returns:
            Dicionário com métricas de estabilidade
        """
        stability_metrics = {
            "kmeans": {"label_consistency": [], "center_variation": []},
            "rskc": {
                "label_consistency": [],
                "center_variation": [],
                "outlier_consistency": [],
            },
        }

        # Executar múltiplas vezes
        for i in range(n_runs):
            # K-means
            kmeans_i = KMeans(
                n_clusters=self.n_clusters, random_state=self.random_state + i
            )
            kmeans_i.fit(self.X)

            # RSKC
            rskc_i = RobustSparseKMeans(
                n_clusters=self.n_clusters,
                # alpha=0.1,
                random_state=self.random_state + i,
            )
            rskc_i.fit(self.X, self.feature_names)

            # Calcular consistência com primeira execução
            if i > 0:
                # K-means stability
                stability_metrics["kmeans"]["label_consistency"].append(
                    adjusted_rand_score(self.kmeans.labels_, kmeans_i.labels_)
                )
                stability_metrics["kmeans"]["center_variation"].append(
                    np.mean(
                        np.abs(self.kmeans.cluster_centers_ - kmeans_i.cluster_centers_)
                    )
                )

                # RSKC stability
                stability_metrics["rskc"]["label_consistency"].append(
                    adjusted_rand_score(self.rskc.labels_, rskc_i.labels_)
                )
                stability_metrics["rskc"]["center_variation"].append(
                    np.mean(
                        np.abs(self.rskc.cluster_centers_ - rskc_i.cluster_centers_)
                    )
                )
                stability_metrics["rskc"]["outlier_consistency"].append(
                    len(set(self.rskc.outliers_).intersection(rskc_i.outliers_))
                    / len(set(self.rskc.outliers_).union(rskc_i.outliers_))
                )

        return stability_metrics

    def create_stability_visualization(self, stability_metrics: Dict) -> go.Figure:
        """
        Cria visualização da análise de estabilidade.
        """
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Consistência dos Rótulos",
                "Variação dos Centros",
                "Consistência dos Outliers (RSKC)",
                "Resumo de Estabilidade",
            ],
        )

        # Plotar consistência dos rótulos
        for method in ["kmeans", "rskc"]:
            fig.add_trace(
                go.Box(
                    y=stability_metrics[method]["label_consistency"],
                    name=method.upper(),
                    boxpoints="all",
                ),
                row=1,
                col=1,
            )

        # Plotar variação dos centros
        for method in ["kmeans", "rskc"]:
            fig.add_trace(
                go.Box(
                    y=stability_metrics[method]["center_variation"],
                    name=method.upper(),
                    boxpoints="all",
                ),
                row=1,
                col=2,
            )

        # Plotar consistência dos outliers (RSKC)
        fig.add_trace(
            go.Box(
                y=stability_metrics["rskc"]["outlier_consistency"],
                name="RSKC Outliers",
                boxpoints="all",
            ),
            row=2,
            col=1,
        )

        # Criar resumo de estabilidade
        stability_scores = {
            "K-means": np.mean(stability_metrics["kmeans"]["label_consistency"]),
            "RSKC": np.mean(stability_metrics["rskc"]["label_consistency"]),
        }

        fig.add_trace(
            go.Bar(
                x=list(stability_scores.keys()),
                y=list(stability_scores.values()),
                name="Estabilidade Geral",
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            height=800,
            title="Análise de Estabilidade dos Métodos",
            showlegend=True,
            template="plotly_white",
        )

        return fig

    def compare_cluster_interpretability(self) -> pd.DataFrame:
        """
        Compara a interpretabilidade dos clusters entre os métodos.
        """
        interpretability = pd.DataFrame()

        for cluster in range(self.n_clusters):
            # K-means analysis
            kmeans_mask = self.kmeans.labels_ == cluster
            kmeans_center = self.kmeans.cluster_centers_[cluster]

            # RSKC analysis
            rskc_mask = self.rskc.labels_ == cluster
            rskc_center = self.rskc.cluster_centers_[cluster]

            # Compare characteristic features
            for method, mask, center in [
                ("K-means", kmeans_mask, kmeans_center),
                ("RSKC", rskc_mask, rskc_center),
            ]:
                cluster_size = np.sum(mask)
                if cluster_size > 0:
                    # Calculate feature contributions
                    contributions = []
                    for i, feature in enumerate(self.feature_names):
                        mean_value = np.mean(self.X[mask, i])
                        std_value = np.std(self.X[mask, i])
                        center_value = center[i]

                        contributions.append(
                            {
                                "feature": feature,
                                "mean": mean_value,
                                "std": std_value,
                                "center_dist": abs(center_value - mean_value),
                            }
                        )

                    # Sort by contribution
                    contributions.sort(key=lambda x: x["center_dist"], reverse=True)

                    # Add to interpretability DataFrame
                    for i, contrib in enumerate(contributions[:3]):
                        interpretability.loc[
                            f"{method} Cluster {cluster} - Feature {i+1}", "Feature"
                        ] = contrib["feature"]
                        interpretability.loc[
                            f"{method} Cluster {cluster} - Feature {i+1}", "Valor Médio"
                        ] = contrib["mean"]
                        interpretability.loc[
                            f"{method} Cluster {cluster} - Feature {i+1}",
                            "Desvio Padrão",
                        ] = contrib["std"]

        return interpretability


def generate_detailed_report(
    comparison: ClusteringComparison, stability_metrics: Optional[Dict] = None
) -> str:
    """
    Gera um relatório textual comparando os resultados dos dois métodos.

    Args:
        comparison: Objeto ClusteringComparison após ajuste dos modelos

    Returns:
        Relatório em formato markdown
    """
    metrics = comparison.comparison_metrics

    report = """
    # Análise Comparativa: K-means vs RSKC
    
    ## 1. Visão Geral
    
    ### Performance Computacional
    - K-means: {kmeans_iter} iterações
    - RSKC: {rskc_iter} iterações
    
    ### Concordância entre Métodos
    - Score de concordância: {agreement:.2f}
    
    ## 2. Análise de Features
    
    ### Features mais importantes:
    
    K-means top 3:
    {kmeans_top_features}
    
    RSKC top 3:
    {rskc_top_features}
    
    ## 3. Distribuição dos Clusters
    
    ### K-means
    - Número total de pontos: {kmeans_total}
    - Distribuição média por cluster: {kmeans_avg:.1f}
    
    ### RSKC
    - Número total de pontos (sem outliers): {rskc_total}
    - Outliers identificados: {rskc_outliers}
    - Distribuição média por cluster: {rskc_avg:.1f}
    
    ## 4. Conclusões
    
    {conclusions}
    """.format(
        kmeans_iter=metrics["kmeans"]["n_iterations"],
        rskc_iter=metrics["rskc"]["n_iterations"],
        agreement=metrics["cluster_agreement"],
        kmeans_top_features="\n".join(
            f"- {f}: {v:.3f}"
            for f, v in sorted(
                metrics["kmeans"]["feature_importance"].items(),
                key=lambda x: x[1],
                reverse=True,
            )[:3]
        ),
        rskc_top_features="\n".join(
            f"- {f}: {v:.3f}"
            for f, v in sorted(
                metrics["rskc"]["feature_importance"].items(),
                key=lambda x: x[1],
                reverse=True,
            )[:3]
        ),
        kmeans_total=len(metrics["kmeans"]["labels"]),
        kmeans_avg=len(metrics["kmeans"]["labels"]) / comparison.n_clusters,
        rskc_total=len(metrics["rskc"]["labels"]) - metrics["rskc"]["n_outliers"],
        rskc_outliers=metrics["rskc"]["n_outliers"],
        rskc_avg=(len(metrics["rskc"]["labels"]) - metrics["rskc"]["n_outliers"])
        / comparison.n_clusters,
        conclusions=_generate_conclusions(metrics),
    )
    # ... [implementação anterior do relatório] ...

    # Adicionar análise de estabilidade se disponível
    if stability_metrics:
        report += "\n## 5. Análise de Estabilidade\n\n"
        report += "### K-means:\n"
        report += f"- Consistência média dos rótulos: {np.mean(stability_metrics['kmeans']['label_consistency']):.3f}\n"
        report += f"- Variação média dos centros: {np.mean(stability_metrics['kmeans']['center_variation']):.3f}\n\n"

        report += "### RSKC:\n"
        report += f"- Consistência média dos rótulos: {np.mean(stability_metrics['rskc']['label_consistency']):.3f}\n"
        report += f"- Variação média dos centros: {np.mean(stability_metrics['rskc']['center_variation']):.3f}\n"
        report += f"- Consistência média dos outliers: {np.mean(stability_metrics['rskc']['outlier_consistency']):.3f}\n"

    return report


def generate_comparison_report(comparison: ClusteringComparison) -> str:
    """
    Gera um relatório textual comparando os resultados dos dois métodos.

    Args:
        comparison: Objeto ClusteringComparison após ajuste dos modelos

    Returns:
        Relatório em formato markdown
    """
    metrics = comparison.comparison_metrics

    report = """
    # Análise Comparativa: K-means vs RSKC
    
    ## 1. Visão Geral
    
    ### Performance Computacional
    - K-means: {kmeans_iter} iterações
    - RSKC: {rskc_iter} iterações
    
    ### Concordância entre Métodos
    - Score de concordância: {agreement:.2f}
    
    ## 2. Análise de Features
    
    ### Features mais importantes:
    
    K-means top 3:
    {kmeans_top_features}
    
    RSKC top 3:
    {rskc_top_features}
    
    ## 3. Distribuição dos Clusters
    
    ### K-means
    - Número total de pontos: {kmeans_total}
    - Distribuição média por cluster: {kmeans_avg:.1f}
    
    ### RSKC
    - Número total de pontos (sem outliers): {rskc_total}
    - Outliers identificados: {rskc_outliers}
    - Distribuição média por cluster: {rskc_avg:.1f}
    
    ## 4. Conclusões
    
    {conclusions}
    """.format(
        kmeans_iter=metrics["kmeans"]["n_iterations"],
        rskc_iter=metrics["rskc"]["n_iterations"],
        agreement=metrics["cluster_agreement"],
        kmeans_top_features="\n".join(
            f"- {f}: {v:.3f}"
            for f, v in sorted(
                metrics["kmeans"]["feature_importance"].items(),
                key=lambda x: x[1],
                reverse=True,
            )[:3]
        ),
        rskc_top_features="\n".join(
            f"- {f}: {v:.3f}"
            for f, v in sorted(
                metrics["rskc"]["feature_importance"].items(),
                key=lambda x: x[1],
                reverse=True,
            )[:3]
        ),
        kmeans_total=len(metrics["kmeans"]["labels"]),
        kmeans_avg=len(metrics["kmeans"]["labels"]) / comparison.n_clusters,
        rskc_total=len(metrics["rskc"]["labels"]) - metrics["rskc"]["n_outliers"],
        rskc_outliers=metrics["rskc"]["n_outliers"],
        rskc_avg=(len(metrics["rskc"]["labels"]) - metrics["rskc"]["n_outliers"])
        / comparison.n_clusters,
        conclusions=_generate_conclusions(metrics),
    )

    return report


def _generate_conclusions(metrics: Dict) -> str:
    """
    Gera conclusões baseadas nas métricas comparativas.
    """
    conclusions = []

    # Analisar concordância
    if metrics["cluster_agreement"] > 0.8:
        conclusions.append(
            "Os métodos apresentam alta concordância na estrutura dos clusters."
        )
    elif metrics["cluster_agreement"] > 0.5:
        conclusions.append(
            "Os métodos mostram concordância moderada na estrutura dos clusters."
        )
    else:
        conclusions.append(
            "Os métodos apresentam diferenças significativas na estrutura dos clusters."
        )

    # Analisar outliers
    if metrics["rskc"]["n_outliers"] > 0:
        outlier_pct = (
            metrics["rskc"]["n_outliers"] / len(metrics["rskc"]["labels"])
        ) * 100
        conclusions.append(
            f"O RSKC identificou {outlier_pct:.1f}% dos pontos como outliers, "
            "sugerindo a presença de ruído nos dados."
        )

    # Analisar features
    kmeans_features = set(
        sorted(
            metrics["kmeans"]["feature_importance"].items(),
            key=lambda x: x[1],
            reverse=True,
        )[:3]
    )

    rskc_features = set(
        sorted(
            metrics["rskc"]["feature_importance"].items(),
            key=lambda x: x[1],
            reverse=True,
        )[:3]
    )

    common_features = kmeans_features.intersection(rskc_features)
    if len(common_features) > 0:
        conclusions.append(
            f"Ambos os métodos identificaram {len(common_features)} features em comum "
            "entre as mais importantes."
        )

    return "\n\n".join(conclusions)


def analyze_kmeans_features(self, kmeans_result, X, feature_names):
    """
    Analisa a importância das features no resultado do K-means.
    """
    # Criar analisador
    feature_analyzer = KMeansFeatureImportance(
        kmeans_model=kmeans_result, X=X, feature_names=feature_names
    )

    # Calcular importância e análise de separação
    importance_scores = feature_analyzer.calculate_feature_importance()
    separation_analysis = feature_analyzer.analyze_cluster_separation()

    # Criar visualização
    importance_plot = create_feature_importance_visualization(
        importance_scores, separation_analysis
    )

    return importance_plot, importance_scores, separation_analysis


import shap
from sklearn.preprocessing import StandardScaler


class ClusteringShapAnalyzer:
    """
    Implementa análise SHAP para modelos de clustering.

    Esta classe utiliza SHAP values para entender como cada feature
    contribui para a formação dos clusters, permitindo uma interpretação
    mais profunda do comportamento dos algoritmos.
    """

    def __init__(
        self, X: np.ndarray, feature_names: List[str], kmeans_model, rskc_model
    ):
        """
        Inicializa o analisador SHAP para clustering.

        Args:
            X: Dados de entrada
            feature_names: Nomes das features
            kmeans_model: Modelo K-means treinado
            rskc_model: Modelo RSKC treinado
        """
        self.X = X
        self.feature_names = feature_names
        self.kmeans = kmeans_model
        self.rskc = rskc_model
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(X)

    def _cluster_contribution_function(self, model, X_background):
        """
        Cria uma função que calcula a contribuição para os clusters.

        Esta função é necessária para o SHAP, pois ele precisa de uma
        função que mapeie features para um valor numérico que represente
        a contribuição para a classificação.
        """

        def contribution(X):
            # Calcular distâncias aos centróides
            distances = np.zeros((X.shape[0], model.n_clusters))
            for k in range(model.n_clusters):
                diff = X - model.cluster_centers_[k]
                if hasattr(model, "weights_"):  # Para RSKC
                    distances[:, k] = np.sum(model.weights_ * diff**2, axis=1)
                else:  # Para K-means
                    distances[:, k] = np.sum(diff**2, axis=1)

            # Retornar o negativo da distância mínima (maior valor = maior contribuição)
            return -np.min(distances, axis=1)

        return contribution

    def calculate_shap_values(self) -> Dict:
        """
        Calcula SHAP values para ambos os modelos.

        Returns:
            Dicionário contendo SHAP values e análises relacionadas
        """
        results = {}

        # Calcular SHAP values para K-means
        kmeans_explainer = shap.KernelExplainer(
            self._cluster_contribution_function(self.kmeans, self.X_scaled),
            self.X_scaled,
        )
        kmeans_shap_values = kmeans_explainer.shap_values(self.X_scaled)

        # Calcular SHAP values para RSKC
        rskc_explainer = shap.KernelExplainer(
            self._cluster_contribution_function(self.rskc, self.X_scaled), self.X_scaled
        )
        rskc_shap_values = rskc_explainer.shap_values(self.X_scaled)

        # Calcular importância global das features
        results["kmeans"] = {
            "shap_values": kmeans_shap_values,
            "feature_importance": np.abs(kmeans_shap_values).mean(axis=0),
            "explainer": kmeans_explainer,
        }

        results["rskc"] = {
            "shap_values": rskc_shap_values,
            "feature_importance": np.abs(rskc_shap_values).mean(axis=0),
            "explainer": rskc_explainer,
        }

        return results

    def create_shap_visualizations(self, shap_results: Dict) -> Dict[str, go.Figure]:
        """
        Cria visualizações comparativas dos SHAP values.
        """
        visualizations = {}

        # 1. Comparação de Importância Global das Features
        fig_importance = self._create_shap_importance_comparison(shap_results)
        visualizations["feature_importance"] = fig_importance

        # 2. SHAP Summary Plots
        fig_summary = self._create_shap_summary_comparison(shap_results)
        visualizations["summary_plots"] = fig_summary

        # 3. Interação entre Features
        fig_interaction = self._create_shap_interaction_comparison(shap_results)
        visualizations["feature_interaction"] = fig_interaction

        return visualizations

    def _create_shap_importance_comparison(self, shap_results: Dict) -> go.Figure:
        """
        Cria visualização comparativa da importância das features via SHAP.
        """
        fig = make_subplots(
            rows=1, cols=2, subplot_titles=["K-means SHAP Values", "RSKC SHAP Values"]
        )

        for idx, method in enumerate(["kmeans", "rskc"], 1):
            importance = pd.Series(
                shap_results[method]["feature_importance"], index=self.feature_names
            ).sort_values(ascending=True)

            fig.add_trace(
                go.Bar(
                    y=importance.index,
                    x=importance.values,
                    orientation="h",
                    name=method.upper(),
                    marker_color=(
                        "rgb(55, 83, 109)"
                        if method == "kmeans"
                        else "rgb(26, 118, 255)"
                    ),
                ),
                row=1,
                col=idx,
            )

        fig.update_layout(
            title="Comparação de Importância das Features (SHAP)",
            height=600,
            showlegend=True,
            template="plotly_white",
        )

        return fig

    def _create_shap_summary_comparison(self, shap_results: Dict) -> go.Figure:
        """
        Cria uma visualização comparativa detalhada dos SHAP values para K-means e RSKC.

        Esta função gera um gráfico que mostra como cada feature contribui para as
        decisões de clustering em ambos os modelos, permitindo uma comparação direta
        da importância e do impacto das features.

        Args:
            shap_results: Dicionário contendo os resultados SHAP para ambos os modelos

        Returns:
            Figura Plotly com os summary plots comparativos
        """
        # Criar figura com subplots para comparação lado a lado
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "K-means SHAP Summary",
                "RSKC SHAP Summary",
                "K-means Feature Impact",
                "RSKC Feature Impact",
            ],
            vertical_spacing=0.2,
            horizontal_spacing=0.1,
        )

        # Processar cada modelo
        for idx, method in enumerate(["kmeans", "rskc"]):
            # Obter SHAP values para o método atual
            shap_values = shap_results[method]["shap_values"]

            # Calcular estatísticas para cada feature
            feature_stats = []
            for i, feature in enumerate(self.feature_names):
                values = shap_values[:, i]
                feature_stats.append(
                    {
                        "feature": feature,
                        "mean_abs_shap": np.mean(np.abs(values)),
                        "mean_shap": np.mean(values),
                        "std_shap": np.std(values),
                        "max_shap": np.max(values),
                        "min_shap": np.min(values),
                    }
                )

            # Ordenar features por importância absoluta
            feature_stats.sort(key=lambda x: x["mean_abs_shap"], reverse=True)

            # Criar violin plot para distribuição dos SHAP values
            for feat_stat in feature_stats:
                feature_idx = self.feature_names.index(feat_stat["feature"])

                # Adicionar violin plot
                fig.add_trace(
                    go.Violin(
                        y=shap_values[:, feature_idx],
                        name=feat_stat["feature"],
                        side="positive",
                        line_color="rgba(0,0,0,0)",
                        fillcolor=f"rgba({50+idx*100},83,109,0.5)",
                        showlegend=False,
                        orientation="h",
                        points=False,
                    ),
                    row=1,
                    col=idx + 1,
                )

            # Criar beeswarm plot para valores de impacto
            for feat_idx, feat_stat in enumerate(feature_stats):
                feature_idx = self.feature_names.index(feat_stat["feature"])
                values = shap_values[:, feature_idx]

                # Adicionar scatter plot
                fig.add_trace(
                    go.Scatter(
                        x=values,
                        y=[feat_idx] * len(values),
                        mode="markers",
                        name=feat_stat["feature"],
                        marker=dict(
                            size=5,
                            color=self.X_scaled[:, feature_idx],
                            colorscale="RdBu",
                            showscale=feat_idx == 0,
                            opacity=0.7,
                        ),
                        showlegend=False,
                    ),
                    row=2,
                    col=idx + 1,
                )

        # Configurar layouts dos subplots
        for i in range(1, 3):
            for j in range(1, 3):
                # Configurar eixos
                fig.update_xaxes(title="SHAP value" if i == 2 else None, row=i, col=j)

                # Configurar rótulos das features
                fig.update_yaxes(
                    title="Features" if j == 1 else None,
                    ticktext=self.feature_names,
                    tickvals=list(range(len(self.feature_names))),
                    row=i,
                    col=j,
                )

        # Atualizar layout geral
        fig.update_layout(
            title={
                "text": "Análise Detalhada do Impacto das Features",
                "y": 0.95,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
            },
            height=1000,
            template="plotly_white",
            showlegend=False,
        )

        # Adicionar anotações explicativas
        fig.add_annotation(
            text=(
                "Valores positivos (vermelho) indicam maior contribuição para o cluster<br>"
                "Valores negativos (azul) indicam menor contribuição"
            ),
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.1,
            showarrow=False,
            font=dict(size=12),
        )

        return fig

    def analyze_feature_interactions(self, shap_results: Dict) -> pd.DataFrame:
        """
        Analisa interações entre features usando SHAP values.
        """
        interaction_analysis = pd.DataFrame(
            index=self.feature_names, columns=["K-means", "RSKC"]
        )

        for method in ["kmeans", "rskc"]:
            shap_values = shap_results[method]["shap_values"]

            # Calcular matriz de interação
            interaction_matrix = np.zeros(
                (len(self.feature_names), len(self.feature_names))
            )

            for i in range(len(self.feature_names)):
                for j in range(len(self.feature_names)):
                    if i != j:
                        interaction = np.abs(
                            shap_values[:, i] * shap_values[:, j]
                        ).mean()
                        interaction_matrix[i, j] = interaction

            # Calcular score de interação para cada feature
            interaction_scores = interaction_matrix.sum(axis=1)
            interaction_analysis[method.upper()] = interaction_scores

        return interaction_analysis

    def _create_shap_interaction_comparison(self, shap_results: Dict) -> go.Figure:
        """
        Cria uma visualização comparativa das interações entre features baseada em SHAP values.
        
        Esta função analisa como as features trabalham em conjunto para influenciar as 
        decisões de clustering, revelando padrões complexos de interação que podem não 
        ser evidentes na análise individual das features.
        
        Args:
            shap_results: Dicionário contendo os resultados SHAP para ambos os modelos
            
        Returns:
            Figura Plotly com matrizes de interação comparativas
        """
        # Criar figura com subplots para K-means e RSKC
        fig = make_subplots(
            rows=1, 
            cols=2,
            subplot_titles=[
                'Interações entre Features - K-means',
                'Interações entre Features - RSKC'
            ],
            horizontal_spacing=0.15
        )
        
        # Processar cada modelo separadamente
        for idx, method in enumerate(['kmeans', 'rskc'], 1):
            # Obter SHAP values do modelo atual
            shap_values = shap_results[method]['shap_values']
            
            # Calcular matriz de interação
            n_features = len(self.feature_names)
            interaction_matrix = np.zeros((n_features, n_features))
            
            # Calcular interações usando produtos dos SHAP values
            for i in range(n_features):
                for j in range(n_features):
                    if i != j:
                        # Calculamos a interação como a média do produto dos SHAP values
                        # Isso captura como as features trabalham juntas
                        interaction = np.abs(
                            shap_values[:, i] * shap_values[:, j]
                        ).mean()
                        
                        # Normalizamos pelo máximo para ter valores entre 0 e 1
                        interaction_matrix[i, j] = interaction
            
            # Normalizar matriz de interação
            if interaction_matrix.max() > 0:
                interaction_matrix = interaction_matrix / interaction_matrix.max()
            
            # Criar heatmap para o modelo atual
            fig.add_trace(
                go.Heatmap(
                    z=interaction_matrix,
                    x=self.feature_names,
                    y=self.feature_names,
                    colorscale='Viridis',
                    colorbar=dict(
                        title='Força da Interação',
                        titleside='right',
                        x=1.15 if idx == 2 else None
                    ),
                    hoverongaps=False,
                    hovertemplate=(
                        'Feature 1: %{y}<br>' +
                        'Feature 2: %{x}<br>' +
                        'Força da Interação: %{z:.3f}<br>' +
                        '<extra></extra>'
                    )
                ),
                row=1,
                col=idx
            )
            
            # Adicionar anotações para interações mais fortes
            threshold = np.percentile(interaction_matrix[interaction_matrix > 0], 90)
            for i in range(n_features):
                for j in range(n_features):
                    if i != j and interaction_matrix[i, j] > threshold:
                        fig.add_annotation(
                            text='★',
                            x=j,
                            y=i,
                            xref=f'x{idx}',
                            yref=f'y{idx}',
                            showarrow=False,
                            font=dict(size=12, color='white')
                        )
        
        # Atualizar layout para melhor visualização
        fig.update_layout(
            title={
                'text': 'Análise de Interações entre Features',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            height=700,
            template='plotly_white',
            showlegend=False
        )
        
        # Adicionar anotações explicativas
        fig.add_annotation(
            text=(
                "★ indica interações particularmente fortes<br>" +
                "Cores mais escuras representam interações mais intensas"
            ),
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.15,
            showarrow=False,
            font=dict(size=12)
        )
        
        # Adicionar interpretação adicional
        strong_interactions = {
            'kmeans': [],
            'rskc': []
        }
        
        for method_idx, method in enumerate(['kmeans', 'rskc']):
            interaction_matrix = np.array([
                [
                    np.abs(shap_results[method]['shap_values'][:, i] * 
                        shap_results[method]['shap_values'][:, j]).mean()
                    for j in range(len(self.feature_names))
                ]
                for i in range(len(self.feature_names))
            ])
            
            # Encontrar pares de features com interações fortes
            threshold = np.percentile(interaction_matrix[interaction_matrix > 0], 90)
            for i in range(len(self.feature_names)):
                for j in range(i + 1, len(self.feature_names)):
                    if interaction_matrix[i, j] > threshold:
                        strong_interactions[method].append(
                            (self.feature_names[i], self.feature_names[j])
                        )
        
        # Adicionar texto interpretativo
        interpretation_text = (
            "Interações Significativas:<br><br>" +
            "<b>K-means:</b><br>" +
            "<br>".join([f"• {f1} ↔ {f2}" for f1, f2 in strong_interactions['kmeans']]) +
            "<br><br><b>RSKC:</b><br>" +
            "<br>".join([f"• {f1} ↔ {f2}" for f1, f2 in strong_interactions['rskc']])
        )
        
        fig.add_annotation(
            text=interpretation_text,
            xref="paper",
            yref="paper",
            x=1.3,
            y=0.5,
            showarrow=False,
            align='left',
            font=dict(size=12)
        )
        
        return fig

def analyze_clustering_differences(
    kmeans_model, rskc_model, X: np.ndarray, feature_names: List[str]
) -> Tuple[Dict, pd.DataFrame]:
    """
    Realiza uma análise aprofundada das diferenças entre K-means e RSKC.

    Esta função combina múltiplas métricas e análises para entender como
    os dois métodos diferem em sua abordagem para clustering.
    """
    # Inicializar analisador SHAP
    shap_analyzer = ClusteringShapAnalyzer(X, feature_names, kmeans_model, rskc_model)
    shap_results = shap_analyzer.calculate_shap_values()

    # Analisar interações entre features
    interaction_analysis = shap_analyzer.analyze_feature_interactions(shap_results)

    # Criar visualizações
    visualizations = shap_analyzer.create_shap_visualizations(shap_results)

    # Compilar resultados
    analysis_results = {
        "shap_values": shap_results,
        "visualizations": visualizations,
        "interaction_analysis": interaction_analysis,
    }

    # Criar tabela comparativa
    comparison_table = pd.DataFrame(
        index=feature_names,
        columns=[
            "K-means Importância",
            "RSKC Importância",
            "K-means SHAP",
            "RSKC SHAP",
            "Diferença Relativa",
        ],
    )

    # Preencher tabela
    for feature in feature_names:
        kmeans_imp = shap_results["kmeans"]["feature_importance"][
            feature_names.index(feature)
        ]
        rskc_imp = shap_results["rskc"]["feature_importance"][
            feature_names.index(feature)
        ]

        comparison_table.loc[feature, "K-means Importância"] = kmeans_imp
        comparison_table.loc[feature, "RSKC Importância"] = rskc_imp
        comparison_table.loc[feature, "K-means SHAP"] = np.abs(
            shap_results["kmeans"]["shap_values"]
        ).mean(axis=0)[feature_names.index(feature)]
        comparison_table.loc[feature, "RSKC SHAP"] = np.abs(
            shap_results["rskc"]["shap_values"]
        ).mean(axis=0)[feature_names.index(feature)]
        comparison_table.loc[feature, "Diferença Relativa"] = (
            rskc_imp - kmeans_imp
        ) / ((rskc_imp + kmeans_imp) / 2)

    return analysis_results, comparison_table
