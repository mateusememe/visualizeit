import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import networkx as nx
from unidecode import unidecode

# Load and clean the data
@st.cache_data
def load_data():
    # df = pd.read_csv('datasets/acidentes_ferroviarios_dez_2020_jul_2024.csv', sep=';')
    df = pd.read_csv('acidents_ferroviarios_2004_2024.csv', sep=';')
    df['Data_Ocorrencia'] = pd.to_datetime(df['Data_Ocorrencia'], format='%d/%m/%y')

    # Clean and convert Quilômetro_Inicial and Quilômetro_Final
    df['Quilômetro_Inicial'] = pd.to_numeric(df['Quilômetro_Inicial'].replace(',', '.'), errors='coerce')
    df['Quilômetro_Final'] = pd.to_numeric(df['Quilômetro_Final'].replace(',', '.'), errors='coerce')
    df['Estação_Anterior'] = df['Estação_Anterior'].apply(unidecode)
    df['Estação_Posterior'] = df['Estação_Posterior'].apply(unidecode)
    return df

df = load_data()

# Set page title
st.title('Visualização de Acidentes Ferroviários - ANTT')

# Sidebar for filtering
st.sidebar.header('Filtros')
concessionaria = st.sidebar.multiselect('Concessionária', options=df['Concessionaria'].unique())
uf = st.sidebar.multiselect('UF', options=df['UF'].unique())

# Apply filters
if concessionaria:
    df = df[df['Concessionaria'].isin(concessionaria)]
if uf:
    df = df[df['UF'].isin(uf)]

# Calculate the number of accidents per municipality
accidents_per_municipality = df.groupby('Municipio').size().to_dict()

# New Graph Visualization
st.header('Grafo de Estações e Acidentes')

# Create a graph
G = nx.Graph()

# Add nodes and edges
for _, row in df.iterrows():
    G.add_node(row['Estação_Anterior'], bipartite=0)
    G.add_node(row['Estação_Posterior'], bipartite=0)
    G.add_node(row['Municipio'], bipartite=1)
    G.add_edge(row['Estação_Anterior'], row['Municipio'])
    G.add_edge(row['Estação_Posterior'], row['Municipio'])
    municipality = row['Municipio']
    G.add_node(municipality, bipartite=1, accidents=accidents_per_municipality[municipality])

# Get position layout for nodes
pos = nx.spring_layout(G)

# Create edge trace
edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines')

# Create node trace
node_x = []
node_y = []
for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    hoverinfo='text',
    marker=dict(
        showscale=True,
        colorscale='YlGnBu',
        reversescale=True,
        color=[],
        size=10,
        colorbar=dict(
            thickness=15,
            title='Conexões',
            xanchor='left',
            titleside='right'
        ),
        line_width=2))

# Color node points by the number of connections
node_adjacencies = []
node_text = []
for node, adjacencies in enumerate(G.adjacency()):
    node_adjacencies.append(len(adjacencies[1]))
    node_text.append(f'{adjacencies[0]}<br># de conexões: {len(adjacencies[1])}')

node_trace.marker.color = node_adjacencies
node_trace.text = node_text

# Create the figure
fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='Grafo de Estações e Locais de Acidentes',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )

st.plotly_chart(fig)

# Display raw data
if st.checkbox('Mostrar dados brutos'):
    st.subheader('Dados Brutos')
    st.write(df)
