# Visualize It

Este projeto demonstra como criar uma aplicação web interativa para visualização de clusterização de dados usando Streamlit e scikit-learn.

## Funcionalidades

- Geração de dados aleatórios em 2D
- Clusterização usando o algoritmo K-means
- Visualização interativa dos clusters
- Ajuste do número de clusters através de um slider
- Exibição dos dados em uma tabela

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
     myenv\Scripts\activate
     ```
   - No macOS e Linux:
     ```bash
     source myenv/bin/activate
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
- `requirements.txt`: Lista de dependências do projeto
- `README.md`: Este arquivo

## Contribuindo

Contribuições são bem-vindas! Por favor, sinta-se à vontade para submeter um Pull Request.

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo [LICENSE.md](LICENSE.md) para detalhes.
