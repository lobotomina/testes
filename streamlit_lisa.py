import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
from libpysal.weights import Queen
from esda.moran import Moran_Local
import tempfile
import os
from io import BytesIO
from pathlib import Path

# Configuração da página
st.set_page_config(
    page_title="Análise Espacial LISA - Abandono Escolar", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2 rem;
        font-family: helvetica, sans-serif;
        font-weight: bold;
        color: #468A9A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .data-info {
        background-color: #EEEEEE;
        border: 1px solid #EEEEEE;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    h4 {
        color: #468A9A;
        font-family: helvetica, sans-serif;
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<h1 class="main-header">Análise Espacial LISA - Abandono Escolar no Ensino Médio</h1>', unsafe_allow_html=True)

# Banner informativo
st.markdown("""
<div class="data-info">
    <h4>Sobre: </h4>
    <p>Aplicativo desenvolvido para análise Análise Espacial LISA - Abandono Escolar no Ensino Médio por ano.
    Trata-se de uma etapa incial de pesquisa acadêmica desenvolvida no conexto da disciplina 'Machine Learning' pelos alunos:  Marcelo Elias, Jane Adriana e Paula Fernanda do PPGTD/UFT.
    Para o desenvolvimento foi utilizado apoio de ferramentas de inteligência artificial generativa com a finalidade precípua de ajuste e correção de códigos. Na verdade, não sabemos ao certo até que ponto ajudou
    ou atrapalhou, mas ainda assim acreditamos ter sido válido, sobretudo pelos efeitos especiais e detalhes estéticos adicionados sem solicitação.</p>
</div>
""", unsafe_allow_html=True)

# Cache para carregamento dos dados
@st.cache_data
def load_data():
    """Carrega os dados dos caminhos relativos"""
    try:
        # Definir caminhos relativos
        base_path = Path(".")  # Diretório atual
        abandono_path = base_path / "data" / "txabandono-municipios.xlsx"
        municipios_path = base_path / "data" / "municipios.csv"
        
        # Verificar se os arquivos existem
        if not abandono_path.exists():
            st.error(f"❌ Arquivo não encontrado: {abandono_path}")
            st.info("💡 Certifique-se de que o arquivo está em: data/txabandono-municipios.xlsx")
            return None, None, False
            
        if not municipios_path.exists():
            st.error(f"❌ Arquivo não encontrado: {municipios_path}")
            st.info("💡 Certifique-se de que o arquivo está em: data/municipios.csv")
            return None, None, False
        
        # Carregar dados de abandono
        st.info(f"Dados separados por ano - selecione o ano desejado na barra lateral")
        df = pd.read_excel(abandono_path)
        
        # Processar dados de abandono
        df["taxa"] = (
            df["Total Abandono no Ens. Médio"]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .replace("--", np.nan)
            .pipe(pd.to_numeric, errors="coerce")
        )
        df = df.dropna(subset=["taxa"])
        df["cod_mun"] = df["cod_mun"].astype(int)
        
        # Carregar dados geográficos
        st.info(f"Os dados do ano de 2021 não devem ser considerados pela existência de inconsistências devido à pandemia de Covid-19.")
        df_geo = pd.read_csv(municipios_path, encoding="latin1")
        
        return df, df_geo, True
        
    except Exception as e:
        st.error(f"❌ Erro ao carregar dados: {e}")
        return None, None, False

@st.cache_data
def calculate_lisa_for_year(df, df_geo, ano):
    """Calcula estatísticas LISA para um ano específico com cache"""
    df_geo = df_geo.rename(columns={"cod_mun": "code_muni"})
    df_geo["code_muni"] = df_geo["code_muni"].astype(int)
    
    # Criar geometria
    gdf_base = gpd.GeoDataFrame(
        df_geo,
        geometry=gpd.points_from_xy(df_geo["longitude"], df_geo["latitude"]),
        crs="EPSG:4326"
    )
    
    # Filtrar dados do ano
    df_ano = df[df["Ano"] == ano].copy()
    if df_ano.empty:
        return None
        
    df_ano["taxa"] = df_ano["taxa"].astype(float)
    df_media = (
        df_ano.groupby("cod_mun")["taxa"]
        .mean()
        .reset_index()
        .rename(columns={"taxa": "taxa_abandono"})
    )

    gdf = gdf_base.merge(df_media, left_on="code_muni", right_on="cod_mun", how="inner")

    if gdf.empty:
        return None

    # Cálculo LISA
    w = Queen.from_dataframe(gdf)
    w.transform = "r"
    y = gdf["taxa_abandono"].values
    lisa = Moran_Local(y, w)

    gdf["LISA_I"] = lisa.Is
    gdf["LISA_p"] = lisa.p_sim
    gdf["LISA_cluster"] = lisa.q
    
    # Mapear clusters para rótulos
    cluster_map = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}
    gdf["LISA_cluster_label"] = gdf["LISA_cluster"].map(cluster_map)
    gdf.loc[gdf["LISA_p"] >= 0.05, "LISA_cluster_label"] = "ns"

    # Juntar Região e UF
    gdf = gdf.merge(df_ano[["cod_mun", "UF", "Região"]].drop_duplicates(),
                    left_on="code_muni", right_on="cod_mun", how="left")

    return gdf

def create_interactive_map(gdf, ano, map_type="cluster"):
    """Cria mapa interativo com Folium"""
    # Cores para clusters
    cores_cluster = {
        "HH": "#d62728",  # Vermelho
        "LL": "#1f77b4",  # Azul
        "LH": "#2ca02c",  # Verde
        "HL": "#ff7f0e",  # Laranja
        "ns": "#7f7f7f"   # Cinza
    }
    
    # Criar mapa base centrado no Brasil
    m = folium.Map(
        location=[-14.2350, -51.9253],  # Centro do Brasil
        zoom_start=4,
        tiles='OpenStreetMap'
    )
    
    # Adicionar pontos ao mapa
    for idx, row in gdf.iterrows():
        if map_type == "cluster":
            color = cores_cluster.get(row["LISA_cluster_label"], "#7f7f7f")
            popup_text = f"""
            <b>{row.get('municipio', 'N/A')}</b><br>
            UF: {row.get('UF', 'N/A')}<br>
            Região: {row.get('Região', 'N/A')}<br>
            Taxa Abandono: {row['taxa_abandono']:.2f}%<br>
            Cluster: {row['LISA_cluster_label']}<br>
            p-valor: {row['LISA_p']:.3f}
            """
        else:  # significance
            color = "#d62728" if row["LISA_p"] < 0.05 else "#7f7f7f"
            popup_text = f"""
            <b>{row.get('municipio', 'N/A')}</b><br>
            UF: {row.get('UF', 'N/A')}<br>
            Região: {row.get('Região', 'N/A')}<br>
            Taxa Abandono: {row['taxa_abandono']:.2f}%<br>
            p-valor: {row['LISA_p']:.3f}<br>
            Significativo: {'Sim' if row['LISA_p'] < 0.05 else 'Não'}
            """
        
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=3,
            popup=popup_text,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7
        ).add_to(m)
    
    return m

def create_plotly_charts(gdf):
    """Cria gráficos interativos com Plotly"""
    # Gráfico de barras - distribuição de clusters
    cluster_counts = gdf['LISA_cluster_label'].value_counts()
    
    fig_bar = px.bar(
        x=cluster_counts.index,
        y=cluster_counts.values,
        title="Distribuição de Clusters LISA",
        labels={'x': 'Tipo de Cluster', 'y': 'Número de Municípios'},
        color=cluster_counts.index,
        color_discrete_map={
            "HH": "#d62728", "LL": "#1f77b4", "LH": "#2ca02c", 
            "HL": "#ff7f0e", "ns": "#7f7f7f"
        }
    )
    fig_bar.update_layout(showlegend=False)
    
    # Histograma das taxas de abandono
    fig_hist = px.histogram(
        gdf, 
        x='taxa_abandono',
        title="Distribuição das Taxas de Abandono",
        labels={'taxa_abandono': 'Taxa de Abandono (%)', 'count': 'Frequência'},
        nbins=30
    )
    
    # Boxplot por região
    if 'Região' in gdf.columns:
        fig_box = px.box(
            gdf, 
            x='Região', 
            y='taxa_abandono',
            title="Taxa de Abandono por Região",
            labels={'taxa_abandono': 'Taxa de Abandono (%)', 'Região': 'Região'}
        )
        fig_box.update_xaxes(tickangle=45)
    else:
        fig_box = None
    
    return fig_bar, fig_hist, fig_box

# Verificar se os arquivos existem
base_path = Path(".")
abandono_path = base_path / "data" / "txabandono-municipios.xlsx"
municipios_path = base_path / "data" / "municipios.csv"

if abandono_path.exists():
    st.sidebar.success("Carregemento efetuado com sucesso")
else:
    st.sidebar.error("❌ txabandono-municipios.xlsx não encontrado")

if municipios_path.exists():
    st.sidebar.success("Municípios do Brasil")
else:
    st.sidebar.error("❌ municipios.csv não encontrado")

# Carregar dados automaticamente
if abandono_path.exists() and municipios_path.exists():
    with st.spinner("🔄 Carregando dados do sistema..."):
        df, df_geo, data_loaded = load_data()

    if data_loaded and df is not None and df_geo is not None:
        # Sidebar com informações dos dados
        st.sidebar.header("Informações:")
        st.sidebar.success(f"""
        **Dados disponíveis**
        
        **Municípios:** {len(df['cod_mun'].unique())}  
        **Anos disponíveis:** {len(df['Ano'].unique())}  
        **Total de registros:** {len(df)}
        **Período:** {df['Ano'].min()} - {df['Ano'].max()}
        """)
        
        # Mostrar anos disponíveis
        anos_disponiveis = sorted(df['Ano'].unique())
        st.sidebar.info(f"**Anos disponíveis:** {', '.join(map(str, anos_disponiveis))}")
        
        # Seleção de ano - PRINCIPAL CONTROLE
        st.sidebar.header("Seleção de Análise")
        ano_selecionado = st.sidebar.selectbox(
            "Selecione o Ano para Análise",
            anos_disponiveis,
            index=len(anos_disponiveis)-1,  # Último ano por padrão
            help="Escolha o ano - exceto 2021"
        )
        
        # Opções de visualização
        st.sidebar.header("Opções de Visualização")
        mostrar_mapas = st.sidebar.checkbox("Mapas interativos", value=True)
        mostrar_graficos = st.sidebar.checkbox("Gráficos estatísticos", value=True)
        mostrar_detalhes = st.sidebar.checkbox("Dados detalhados", value=True)
        
        # Calcular LISA para o ano selecionado
        with st.spinner(f"🧮 Calculando estatísticas LISA para {ano_selecionado}..."):
            gdf = calculate_lisa_for_year(df, df_geo, ano_selecionado)
        
        if gdf is not None:
            # Métricas principais
            st.subheader(f"Resultados da Análise LISA - {ano_selecionado}")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_municipios = len(gdf)
                st.metric("🏘️ Total de Municípios", total_municipios)
            
            with col2:
                significativos = len(gdf[gdf['LISA_p'] < 0.05])
                percentual_sig = (significativos / total_municipios) * 100
                st.metric("✅ Clusters Significativos", f"{significativos} ({percentual_sig:.1f}%)")
            
            with col3:
                taxa_media = gdf['taxa_abandono'].mean()
                st.metric("📈 Taxa Média de Abandono", f"{taxa_media:.2f}%")
            
            with col4:
                moran_i = gdf['LISA_I'].mean()
                st.metric("🔗 Índice de Moran (médio)", f"{moran_i:.3f}")
            
            # Tabs para diferentes visualizações
            tabs_list = []
            if mostrar_mapas:
                tabs_list.extend(["🗺️ Mapa de Clusters", "📊 Mapa de Significância"])
            if mostrar_graficos:
                tabs_list.append("📈 Gráficos Estatísticos")
            if mostrar_detalhes:
                tabs_list.append("📋 Dados Detalhados")
            
            if tabs_list:
                tabs = st.tabs(tabs_list)
                tab_index = 0
                
                if mostrar_mapas:
                    # Tab Mapa de Clusters
                    with tabs[tab_index]:
                        st.subheader(f"Mapa LISA - Clusters - {ano_selecionado}")
                        mapa_cluster = create_interactive_map(gdf, ano_selecionado, "cluster")
                        st_folium(mapa_cluster, width=700, height=500)
                        
                        # Legenda
                        st.markdown("""
                        **Legenda dos Clusters:**
                        - 🔴 **HH (Alto-Alto):** Municípios com alta taxa cercados por municípios com alta taxa
                        - 🔵 **LL (Baixo-Baixo):** Municípios com baixa taxa cercados por municípios com baixa taxa  
                        - 🟢 **LH (Baixo-Alto):** Municípios com baixa taxa cercados por municípios com alta taxa
                        - 🟠 **HL (Alto-Baixo):** Municípios com alta taxa cercados por municípios com baixa taxa
                        - ⚫ **ns (Não significativo):** Sem padrão espacial significativo (p ≥ 0.05)
                        """)
                    tab_index += 1
                    
                    # Tab Mapa de Significância
                    with tabs[tab_index]:
                        st.subheader(f"Mapa LISA - Significância Estatística - {ano_selecionado}")
                        mapa_sig = create_interactive_map(gdf, ano_selecionado, "significance")
                        st_folium(mapa_sig, width=700, height=500)
                        
                        st.markdown("""
                        **Legenda da Significância:**
                        - 🔴 **Significativo (p < 0.05):** Padrão espacial estatisticamente significativo
                        - ⚫ **Não significativo (p ≥ 0.05):** Sem padrão espacial significativo
                        """)
                    tab_index += 1
                
                if mostrar_graficos:
                    # Tab Gráficos
                    with tabs[tab_index]:
                        st.subheader("Análise Estatística")
                        
                        fig_bar, fig_hist, fig_box = create_plotly_charts(gdf)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.plotly_chart(fig_bar, use_container_width=True)
                        with col2:
                            st.plotly_chart(fig_hist, use_container_width=True)
                        
                        if fig_box:
                            st.plotly_chart(fig_box, use_container_width=True)
                    tab_index += 1
                
                if mostrar_detalhes:
                    # Tab Dados Detalhados
                    with tabs[tab_index]:
                        st.subheader("Dados Detalhados")
                        
                        # Filtros para os dados
                        col1, col2 = st.columns(2)
                        with col1:
                            cluster_filter = st.multiselect(
                                "Filtrar por tipo de cluster:",
                                options=gdf['LISA_cluster_label'].unique(),
                                default=gdf['LISA_cluster_label'].unique()
                            )
                        with col2:
                            sig_filter = st.selectbox(
                                "Filtrar por significância:",
                                options=["Todos", "Significativos (p < 0.05)", "Não significativos (p ≥ 0.05)"]
                            )
                        
                        # Aplicar filtros
                        gdf_filtered = gdf[gdf['LISA_cluster_label'].isin(cluster_filter)]
                        if sig_filter == "Significativos (p < 0.05)":
                            gdf_filtered = gdf_filtered[gdf_filtered['LISA_p'] < 0.05]
                        elif sig_filter == "Não significativos (p ≥ 0.05)":
                            gdf_filtered = gdf_filtered[gdf_filtered['LISA_p'] >= 0.05]
                        
                        # Mostrar dados
                        colunas_mostrar = ['municipio', 'UF', 'Região', 'taxa_abandono', 
                                         'LISA_cluster_label', 'LISA_I', 'LISA_p']
                        colunas_disponiveis = [col for col in colunas_mostrar if col in gdf_filtered.columns]
                        
                        st.dataframe(
                            gdf_filtered[colunas_disponiveis].round(3),
                            use_container_width=True
                        )
                        
                        # Download dos dados
                        @st.cache_data
                        def convert_df_to_csv(df):
                            return df.to_csv(index=False).encode('utf-8')
                        
                        csv_data = convert_df_to_csv(gdf_filtered[colunas_disponiveis])
                        st.download_button(
                            label="📥 Baixar dados filtrados (CSV)",
                            data=csv_data,
                            file_name=f"lisa_dados_{ano_selecionado}.csv",
                            mime="text/csv"
                        )
            
            # Resumo estatístico sempre visível
            st.subheader("📊 Resumo Estatístico")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Distribuição de Clusters:**")
                cluster_summary = gdf['LISA_cluster_label'].value_counts()
                for cluster, count in cluster_summary.items():
                    percentage = (count / len(gdf)) * 100
                    st.write(f"- **{cluster}**: {count} municípios ({percentage:.1f}%)")
            
            with col2:
                st.write("**Estatísticas da Taxa de Abandono:**")
                st.write(f"- **Média**: {gdf['taxa_abandono'].mean():.2f}%")
                st.write(f"- **Mediana**: {gdf['taxa_abandono'].median():.2f}%")
                st.write(f"- **Desvio padrão**: {gdf['taxa_abandono'].std():.2f}%")
                st.write(f"- **Mínimo**: {gdf['taxa_abandono'].min():.2f}%")
                st.write(f"- **Máximo**: {gdf['taxa_abandono'].max():.2f}%")
        
        else:
            st.error(f"❌ Não foi possível calcular as estatísticas LISA para o ano {ano_selecionado}. Verifique se há dados disponíveis para este ano.")

    else:
        st.error("❌ Erro ao carregar os dados do sistema. Verifique se os arquivos estão no formato correto.")

else:
    st.warning("""
    ⚠️ **Arquivos de dados não encontrados!**
    
    Para usar este aplicativo, você precisa ter a seguinte estrutura de pastas:
    
    ```
    seu_projeto/
    ├── streamlit_lisa_vscode.py  (este arquivo)
    └── data/
        ├── municipios.csv
        └── txabandono-municipios.xlsx
    ```
    
    **Passos para configurar:**
    1. Crie uma pasta chamada `data` no mesmo diretório deste arquivo
    2. Coloque os arquivos `municipios.csv` e `txabandono-municipios.xlsx` dentro da pasta `data`
    3. Execute novamente o aplicativo
    """)
    
    # Mostrar instruções de execução
    st.info("""
    **💡 Como executar este aplicativo:**
    
    1. **Instale as dependências:**
    ```bash
    pip install streamlit pandas numpy geopandas plotly folium streamlit-folium libpysal esda openpyxl
    ```
    
    2. **Execute o aplicativo:**
    ```bash
    streamlit run streamlit_lisa_vscode.py
    ```
    """)
