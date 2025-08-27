import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import libpysal as lps
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
from libpysal.weights import Queen
from esda.moran import Moran_Local
import tempfile
import os
from io import BytesIO
import base64
from pathlib import Path
import logging
from typing import Optional, Tuple

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuração da página
st.set_page_config(
    page_title="Análise Espacial LISA - Abandono Escolar", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constantes
CLUSTER_COLORS = {
    "HH": "#d62728",  # Vermelho
    "LL": "#1f77b4",  # Azul
    "LH": "#2ca02c",  # Verde
    "HL": "#ff7f0e",  # Laranja
    "ns": "#7f7f7f"   # Cinza
}

CLUSTER_LABELS = {
    1: "HH", 2: "LH", 3: "LL", 4: "HL"
}

# CSS customizado (simplificado)
def load_css():
    """Carrega CSS customizado"""
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-container {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .data-info {
            background-color: #e8f4fd;
            border: 1px solid #1f77b4;
            border-radius: 0.5rem;
            padding: 1rem;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

class DataLoader:
    """Classe para gerenciar carregamento de dados"""
    
    @staticmethod
    @st.cache_data
    def load_data() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], bool]:
        """Carrega os dados das planilhas com tratamento de erro robusto"""
        try:
            # Tentar diferentes caminhos para os arquivos
            possible_paths = [
                ("data/txabandono-municipios.xlsx", "data/municipios.csv"),
                ("../data/txabandono-municipios.xlsx", "../data/municipios.csv"),
                ("./data/txabandono-municipios.xlsx", "./data/municipios.csv")
            ]
            
            df, df_geo = None, None
            
            for abandono_path, geo_path in possible_paths:
                try:
                    if Path(abandono_path).exists() and Path(geo_path).exists():
                        df = pd.read_excel(abandono_path)
                        df_geo = pd.read_csv(geo_path, encoding="latin1")
                        logger.info(f"Dados carregados de: {abandono_path}, {geo_path}")
                        break
                except Exception as e:
                    logger.warning(f"Falha ao carregar de {abandono_path}: {e}")
                    continue
            
            if df is None or df_geo is None:
                return None, None, False
            
            # Processar dados de abandono
            df = DataLoader.process_abandono_data(df)
            
            return df, df_geo, True
            
        except Exception as e:
            logger.error(f"Erro geral ao carregar dados: {e}")
            st.error(f"Erro ao carregar dados: {e}")
            return None, None, False
    
    @staticmethod
    def process_abandono_data(df: pd.DataFrame) -> pd.DataFrame:
        """Processa os dados de abandono"""
        df["taxa"] = (
            df["Total Abandono no Ens. Médio"]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .replace("--", np.nan)
            .pipe(pd.to_numeric, errors="coerce")
        )
        df = df.dropna(subset=["taxa"])
        df["cod_mun"] = df["cod_mun"].astype(int)
        return df

class LISACalculator:
    """Classe para cálculos LISA"""
    
    @staticmethod
    @st.cache_data
    def calculate_lisa_for_year(df: pd.DataFrame, df_geo: pd.DataFrame, ano: int) -> Optional[gpd.GeoDataFrame]:
        """Calcula estatísticas LISA para um ano específico"""
        try:
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
                logger.warning(f"Nenhum dado encontrado para o ano {ano}")
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
                logger.warning("GeoDataFrame vazio após merge")
                return None

            # Cálculo LISA com tratamento de erro
            try:
                w = Queen.from_dataframe(gdf)
                w.transform = "r"
                y = gdf["taxa_abandono"].values
                lisa = Moran_Local(y, w)

                gdf["LISA_I"] = lisa.Is
                gdf["LISA_p"] = lisa.p_sim
                gdf["LISA_cluster"] = lisa.q
                
                # Mapear clusters para rótulos
                gdf["LISA_cluster_label"] = gdf["LISA_cluster"].map(CLUSTER_LABELS)
                gdf.loc[gdf["LISA_p"] >= 0.05, "LISA_cluster_label"] = "ns"

                # Juntar Região e UF
                gdf = gdf.merge(df_ano[["cod_mun", "UF", "Região"]].drop_duplicates(),
                                left_on="code_muni", right_on="cod_mun", how="left")

                return gdf
                
            except Exception as e:
                logger.error(f"Erro no cálculo LISA: {e}")
                st.error(f"Erro no cálculo LISA: {e}")
                return None
                
        except Exception as e:
            logger.error(f"Erro geral no cálculo LISA: {e}")
            return None

class MapVisualizer:
    """Classe para visualizações de mapas"""
    
    @staticmethod
    def create_interactive_map(gdf: gpd.GeoDataFrame, ano: int, map_type: str = "cluster") -> folium.Map:
        """Cria mapa interativo com Folium - versão otimizada"""
        # Criar mapa base centrado no Brasil
        m = folium.Map(
            location=[-14.2350, -51.9253],
            zoom_start=4,
            tiles='OpenStreetMap'
        )
        
        # Adicionar pontos ao mapa (otimizado)
        for idx, row in gdf.iterrows():
            if pd.isna(row.geometry.x) or pd.isna(row.geometry.y):
                continue
                
            if map_type == "cluster":
                color = CLUSTER_COLORS.get(row["LISA_cluster_label"], "#7f7f7f")
                popup_text = f"""
                <b>{row.get('municipio', 'N/A')}</b><br>
                UF: {row.get('UF', 'N/A')}<br>
                Taxa Abandono: {row['taxa_abandono']:.2f}%<br>
                Cluster: {row['LISA_cluster_label']}<br>
                p-valor: {row['LISA_p']:.3f}
                """
            else:  # significance
                color = "#d62728" if row["LISA_p"] < 0.05 else "#7f7f7f"
                popup_text = f"""
                <b>{row.get('municipio', 'N/A')}</b><br>
                UF: {row.get('UF', 'N/A')}<br>
                Taxa Abandono: {row['taxa_abandono']:.2f}%<br>
                p-valor: {row['LISA_p']:.3f}<br>
                Significativo: {'Sim' if row['LISA_p'] < 0.05 else 'Não'}
                """
            
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=4,
                popup=popup_text,
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.8,
                weight=1
            ).add_to(m)
        
        return m

class ChartCreator:
    """Classe para criação de gráficos"""
    
    @staticmethod
    def create_plotly_charts(gdf: gpd.GeoDataFrame) -> Tuple[go.Figure, go.Figure, Optional[go.Figure]]:
        """Cria gráficos interativos com Plotly - versão otimizada"""
        # Gráfico de barras - distribuição de clusters
        cluster_counts = gdf['LISA_cluster_label'].value_counts()
        
        fig_bar = px.bar(
            x=cluster_counts.index,
            y=cluster_counts.values,
            title="Distribuição de Clusters LISA",
            labels={'x': 'Tipo de Cluster', 'y': 'Número de Municípios'},
            color=cluster_counts.index,
            color_discrete_map=CLUSTER_COLORS
        )
        fig_bar.update_layout(showlegend=False, height=400)
        
        # Histograma das taxas de abandono
        fig_hist = px.histogram(
            gdf, 
            x='taxa_abandono',
            title="Distribuição das Taxas de Abandono",
            labels={'taxa_abandono': 'Taxa de Abandono (%)', 'count': 'Frequência'},
            nbins=25
        )
        fig_hist.update_layout(height=400)
        
        # Boxplot por região
        fig_box = None
        if 'Região' in gdf.columns and not gdf['Região'].isna().all():
            fig_box = px.box(
                gdf, 
                x='Região', 
                y='taxa_abandono',
                title="Taxa de Abandono por Região",
                labels={'taxa_abandono': 'Taxa de Abandono (%)', 'Região': 'Região'}
            )
            fig_box.update_xaxes(tickangle=45)
            fig_box.update_layout(height=400)
        
        return fig_bar, fig_hist, fig_box

def create_file_uploader():
    """Cria interface para upload de arquivos se os dados não existirem"""
    st.warning("📁 Arquivos de dados não encontrados. Faça upload dos arquivos necessários:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_abandono = st.file_uploader(
            "Upload: txabandono-municipios.xlsx", 
            type=['xlsx', 'xls'],
            help="Arquivo Excel com dados de abandono escolar"
        )
    
    with col2:
        uploaded_geo = st.file_uploader(
            "Upload: municipios.csv", 
            type=['csv'],
            help="Arquivo CSV com dados geográficos dos municípios"
        )
    
    if uploaded_abandono and uploaded_geo:
        try:
            df = pd.read_excel(uploaded_abandono)
            df_geo = pd.read_csv(uploaded_geo, encoding="latin1")
            
            # Processar dados
            df = DataLoader.process_abandono_data(df)
            
            return df, df_geo, True
        except Exception as e:
            st.error(f"Erro ao processar arquivos: {e}")
            return None, None, False
    
    return None, None, False

def main():
    """Função principal da aplicação"""
    load_css()
    
    # Título principal
    st.markdown('<h1 class="main-header">Análise Espacial LISA - Abandono Escolar no Ensino Médio</h1>', unsafe_allow_html=True)
    
    # Banner informativo
    st.markdown("""
    <div class="data-info">
        <h3>📊 Análise de Autocorrelação Espacial</h3>
        <p>Esta aplicação realiza análise LISA (Local Indicators of Spatial Association) 
        para identificar clusters espaciais de abandono escolar no ensino médio.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Carregar dados
    with st.spinner("🔄 Carregando dados do sistema..."):
        df, df_geo, data_loaded = DataLoader.load_data()
    
    # Se não conseguir carregar, oferecer upload
    if not data_loaded:
        df, df_geo, data_loaded = create_file_uploader()
    
    if data_loaded and df is not None and df_geo is not None:
        # Sidebar com informações
        create_sidebar_info(df)
        
        # Seleção de ano
        anos_disponiveis = sorted(df['Ano'].unique())
        ano_selecionado = st.sidebar.selectbox(
            "📅 Selecione o Ano para Análise",
            anos_disponiveis,
            index=len(anos_disponiveis)-1,
            help="Escolha o ano que deseja analisar"
        )
        
        # Opções de visualização
        create_visualization_options()
        
        # Calcular LISA
        with st.spinner(f"Calculando estatísticas LISA para {ano_selecionado}..."):
            gdf = LISACalculator.calculate_lisa_for_year(df, df_geo, ano_selecionado)
        
        if gdf is not None:
            # Mostrar resultados
            display_results(gdf, ano_selecionado)
        else:
            st.error(f"❌ Não foi possível calcular as estatísticas LISA para o ano {ano_selecionado}")
    
    else:
        st.error("❌ Não foi possível carregar os dados necessários.")

def create_sidebar_info(df: pd.DataFrame):
    """Cria informações na sidebar"""
    st.sidebar.header("📊 Informações dos Dados")
    st.sidebar.success(f"""
    **✅ Dados Carregados!**
    
    **Municípios:** {len(df['cod_mun'].unique()):,}  
    **Anos disponíveis:** {len(df['Ano'].unique())}  
    **Registros:** {len(df):,}
    **Período:** {df['Ano'].min()} - {df['Ano'].max()}
    """)

def create_visualization_options():
    """Cria opções de visualização na sidebar"""
    st.sidebar.header("🔍 Opções de Visualização")
    return {
        'mapas': st.sidebar.checkbox("Mostrar mapas interativos", value=True),
        'graficos': st.sidebar.checkbox("Mostrar gráficos estatísticos", value=True),
        'detalhes': st.sidebar.checkbox("Mostrar dados detalhados", value=True)
    }

def display_results(gdf: gpd.GeoDataFrame, ano: int):
    """Exibe os resultados da análise"""
    # Métricas principais
    st.subheader(f"📈 Resultados da Análise LISA - {ano}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Municípios", f"{len(gdf):,}")
    
    with col2:
        significativos = len(gdf[gdf['LISA_p'] < 0.05])
        percentual = (significativos / len(gdf)) * 100
        st.metric("Clusters Significativos", f"{significativos:,} ({percentual:.1f}%)")
    
    with col3:
        st.metric("Taxa Média de Abandono", f"{gdf['taxa_abandono'].mean():.2f}%")
    
    with col4:
        st.metric("Índice de Moran (médio)", f"{gdf['LISA_I'].mean():.3f}")
    
    # Tabs para visualizações
    tab1, tab2, tab3 = st.tabs(["🗺️ Mapas", "📊 Gráficos", "📋 Dados"])
    
    with tab1:
        display_maps(gdf, ano)
    
    with tab2:
        display_charts(gdf)
    
    with tab3:
        display_data_table(gdf, ano)

def display_maps(gdf: gpd.GeoDataFrame, ano: int):
    """Exibe os mapas interativos"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Clusters LISA")
        mapa_cluster = MapVisualizer.create_interactive_map(gdf, ano, "cluster")
        st_folium(mapa_cluster, width=350, height=400)
    
    with col2:
        st.subheader("Significância Estatística")
        mapa_sig = MapVisualizer.create_interactive_map(gdf, ano, "significance")
        st_folium(mapa_sig, width=350, height=400)

def display_charts(gdf: gpd.GeoDataFrame):
    """Exibe os gráficos estatísticos"""
    fig_bar, fig_hist, fig_box = ChartCreator.create_plotly_charts(gdf)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_bar, use_container_width=True)
    with col2:
        st.plotly_chart(fig_hist, use_container_width=True)
    
    if fig_box:
        st.plotly_chart(fig_box, use_container_width=True)

def display_data_table(gdf: gpd.GeoDataFrame, ano: int):
    """Exibe a tabela de dados"""
    st.subheader("Tabela de Dados")
    
    # Filtros
    col1, col2 = st.columns(2)
    with col1:
        cluster_filter = st.multiselect(
            "Filtrar por cluster:",
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
    
    # Colunas para mostrar
    colunas_mostrar = ['municipio', 'UF', 'Região', 'taxa_abandono', 
                     'LISA_cluster_label', 'LISA_I', 'LISA_p']
    colunas_disponiveis = [col for col in colunas_mostrar if col in gdf_filtered.columns]
    
    # Mostrar dados
    st.dataframe(gdf_filtered[colunas_disponiveis].round(3), use_container_width=True)
    
    # Download
    csv_data = gdf_filtered[colunas_disponiveis].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Baixar dados (CSV)",
        data=csv_data,
        file_name=f"lisa_dados_{ano}.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()