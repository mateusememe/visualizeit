# Visualize It

Este projeto utiliza técnicas avançadas de visualização de dados para analisar acidentes ferroviários no Brasil, baseado em dados públicos da ANTT (Agência Nacional de Transportes Terrestres) [disponiveis aqui](https://dados.antt.gov.br/dataset/relatorio-de-acompanhamento-de-acidentes-ferroviarios-raaf).

## Técnicas de Visualização Avançadas

O projeto explora as seguintes técnicas de visualização avançadas:

1. **Mapas Interativos**: Utiliza Plotly Express para criar mapas interativos que mostram a distribuição geográfica dos acidentes.
2. **Gráficos de Barras Dinâmicos**: Apresenta informações sobre acidentes por concessionária e UF.
3. **Gráficos de Linhas Temporais**: Mostra a evolução dos acidentes ao longo do tempo.
4. **Gráficos de Pizza Interativos**: Visualiza a distribuição de causas diretas e natureza dos acidentes.
5. **Clusterização Geoespacial**: Aplica K-means para agrupar acidentes com base em localização e frequência.
6. **Filtros Interativos**: Permite a filtragem dinâmica dos dados por diversos critérios.

## Pré-requisitos

- Python 3.7+
- pip

## Instalação

1. Clone este repositório:
   ```bash
   git clone https://github.com/mateusememe/visualizeit.git
   cd visualizeit
   ```

2. Crie um ambiente virtual:
   ```bash
   python -m venv myenv
   ```

3. Ative o ambiente virtual:
   - No Windows:
     ```bash
     myenv\Scripts\activate # Para ativar
     myenv\bin\deactivate # Para desativar
     ```
   - No macOS e Linux:
     ```bash
     source myenv/bin/activate # Para ativar
     source myenv/bin/deactivate # Para desativar
     ```

4. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

## Uso

Para executar a aplicação:

```bash
streamlit run app.py
```

Abra seu navegador e acesse `http://localhost:8501` para ver a aplicação em execução.

## Estrutura do Projeto

- `app.py`: O código principal da aplicação Streamlit
- `preprocessing_data_including_geolocalization.py`: O código de préprocessamento do csv original para termos as coordenadas dos municipios
- `requirements.txt`: Lista de dependências do projeto
- `datasets/` diretório com os arquivos csv utilizados para visualização
  - `acidentes_ferroviarios_12.2020-07.2024.csv` - base de dados de Dezembro de 2020 á Julho de 2024
  - `acidentes_ferroviarios_2004_2020.csv` - base de dados de 2004 á Novembro de 2020
  - `acidentes_ferroviarios_2004_2024.csv` - base de dados unificado dos casos de 2004 á Julho de 2024
- `README.md`: Este arquivo

## Insights Potenciais

Baseado no código, o projeto permite extrair os seguintes insights:

1. **Distribuição Geográfica**: Identificar áreas de alta concentração de acidentes.
2. **Padrões Temporais**: Analisar tendências de acidentes ao longo do tempo.
3. **Análise por Concessionária**: Comparar o desempenho de segurança entre diferentes operadoras.
4. **Causas Comuns**: Identificar as causas mais frequentes de acidentes.
5. **Natureza dos Acidentes**: Entender os tipos mais comuns de incidentes.
6. **Clusters de Risco**: Identificar regiões com características similares em termos de frequência de acidentes.
7. **Impacto de Mercadorias**: Analisar se certos tipos de mercadorias estão associados a maiores riscos.
8. **Variações Sazonais**: Investigar se há padrões sazonais nos acidentes.
9. **Hotspots**: Identificar municípios ou linhas com frequência anormalmente alta de acidentes.

## Notas Adicionais

- Certifique-se de ter o arquivo CSV com os dados dos acidentes no diretório do projeto.
- As visualizações são interativas, permitindo zoom, hover e seleção de dados.
- O slider para escolha do número de clusters permite uma análise flexível da distribuição geográfica dos acidentes.

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo [LICENSE.md](LICENSE.md) para detalhes.
