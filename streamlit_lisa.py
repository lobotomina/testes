import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import logging
from typing import Optional, Tuple

# Tentar importar dependências geoespaciais com fallbacks
try:
    import geopandas as gpd
    import libpysal as lps
    from libpysal.weights import Queen
    from esda.moran import Moran_Local
    GEOSPATIAL_AVAILABLE = True
except ImportError as e:
    st.error(f"⚠️ Dependências geoespaciais não disponíveis: {e}")
    GEOSPATIAL_AVAILABLE = False

# Tentar importar folium com fallback
try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    st.warning("Folium não disponível - usando visualizações alternativas")

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

CLUSTER_LABELS = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}

CLUSTER_DESCRIPTIONS = {
    "HH": "Alto-Alto: Municípios com alta taxa cercados por municípios com alta taxa",
    "LL": "Baixo-Baixo: Municípios com baixa taxa cercados por municípios com baixa taxa", 
    "LH": "Baixo-Alto: Municípios com baixa taxa cercados por municípios com alta taxa",
    "HL": "Alto-Baixo: Municípios com alta taxa cercados por municípios com baixa taxa",
    "ns": "Não significativo: Sem padrão espacial significativo (p ≥ 0.05)"
}

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
        .warning-box {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 0.5rem;
            padding: 1rem;
            margin: 1rem 0;
        }
        .success-box {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
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
            possible_paths = [
                ("data/txabandono-municipios.xlsx", "data/municipios.csv"),
                ("../data/txabandono-municipios.xlsx", "../data/municipios.csv"),
                ("./data/txabandono-municipios.xlsx", "./data/municipios.csv"),
                ("txabandono-municipios.xlsx", "municipios.csv")
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
            
            df = DataLoader.process_abandono_data(df)
            return df, df_geo, True
            
        except Exception as e:
            logger.error(f"Erro geral ao carregar dados: {e}")
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
    def calculate_lisa_for_year(df: pd.DataFrame, df_geo: pd.DataFrame, ano: int):
        """Calcula estatísticas LISA para um ano específico"""
        if not GEOSPATIAL_AVAILABLE:
            return None
            
        try:
            df_geo = df_geo.rename(columns={"cod_mun": "code_muni"})
            df_geo["code_muni"] = df_geo["code_muni"].astype(int)
            
            gdf_base = gpd.GeoDataFrame(
                df_geo,
                geometry=gpd.points_from_xy(df_geo["longitude"], df_geo["latitude"]),
                crs="EPSG:4326"
            )
            
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

            try:
                w = Queen.from_dataframe(gdf)
                w.transform = "r"
                y = gdf["taxa_abandono"].values
                lisa = Moran_Local(y, w)

                gdf["LISA_I"] = lisa.Is
                gdf["LISA_p"] = lisa.p_sim
                gdf["LISA_cluster"] = lisa.q
                
                gdf["LISA_cluster_label"] = gdf["LISA_cluster"].map(CLUSTER_LABELS)
                gdf.loc[gdf["LISA_p"] >= 0.05, "LISA_cluster_label"] = "ns"

                gdf = gdf.merge(df_ano[["cod_mun", "UF", "Região"]].drop_duplicates(),
                                left_on="code_muni", right_on="cod_mun", how="left")

                return gdf
                
            except Exception as e:
                logger.error(f"Erro no cálculo LISA: {e}")
                return None
                
        except Exception as e:
            logger.error(f"Erro geral no cálculo LISA: {e}")
            return None

class MapVisualizer:
    """Classe para visualizações de mapas - VERSÃO CORRIGIDA"""
    
    @staticmethod
    def create_simple_folium_map(gdf, ano: int, map_type: str = "cluster"):
        """Cria mapa simples com Folium - sem serialização problemática"""
        if not FOLIUM_AVAILABLE:
            return None
            
        try:
            # Criar mapa base
            center_lat = gdf.geometry.y.mean()
            center_lon = gdf.geometry.x.mean()
            
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=5,
                tiles='OpenStreetMap'
            )
            
            # Adicionar pontos (versão simplificada)
            for idx, row in gdf.head(100).iterrows():  # Limitar a 100 pontos para evitar problemas
                if pd.isna(row.geometry.x) or pd.isna(row.geometry.y):
                    continue
                
                if map_type == "cluster" and 'LISA_cluster_label' in gdf.columns:
                    color = CLUSTER_COLORS.get(row["LISA_cluster_label"], "#7f7f7f")
                    popup_text = f"{row.get('municipio', 'N/A')} - {row['LISA_cluster_label']}"
                else:
                    if 'LISA_p' in gdf.columns:
                        color = "#d62728" if row["LISA_p"] < 0.05 else "#7f7f7f"
                        popup_text = f"{row.get('municipio', 'N/A')} - p: {row['LISA_p']:.3f}"
                    else:
                        color = "#1f77b4"
                        popup_text = f"{row.get('municipio', 'N/A')} - Taxa: {row['taxa_abandono']:.2f}%"
                
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=4,
                    popup=folium.Popup(popup_text, max_width=200),
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.7,
                    weight=1
                ).add_to(m)
            
            return m
        except Exception as e:
            logger.error(f"Erro ao criar mapa Folium: {e}")
            return None

class ChartCreator:
    """Classe para criação de gráficos"""
    
    @staticmethod
    def create_plotly_charts(gdf) -> Tuple[go.Figure, go.Figure, Optional[go.Figure], go.Figure]:
        """Cria gráficos interativos com Plotly"""
        # 1. Gráfico de barras - distribuição de clusters
        if 'LISA_cluster_label' in gdf.columns:
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
        else:
            # Gráfico alternativo se não tiver LISA
            fig_bar = px.bar(
                x=['Municípios Analisados'],
                y=[len(gdf)],
                title="Total de Municípios Analisados"
            )
        
        # 2. Histograma das taxas de abandono
        fig_hist = px.histogram(
            gdf, 
            x='taxa_abandono',
            title="Distribuição das Taxas de Abandono",
            labels={'taxa_abandono': 'Taxa de Abandono (%)', 'count': 'Frequência'},
            nbins=25
        )
        fig_hist.update_layout(height=400)
        
        # 3. Boxplot por região
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
        
        # 4. Scatter plot geográfico
        fig_scatter = px.scatter(
            gdf,
            x='longitude' if 'longitude' in gdf.columns else gdf.geometry.x,
            y='latitude' if 'latitude' in gdf.columns else gdf.geometry.y,
            color='LISA_cluster_label' if 'LISA_cluster_label' in gdf.columns else 'taxa_abandono',
            size='taxa_abandono',
            hover_name='municipio' if 'municipio' in gdf.columns else None,
            hover_data=['UF', 'Região', 'taxa_abandono'] if all(col in gdf.columns for col in ['UF', 'Região']) else ['taxa_abandono'],
            title="Distribuição Espacial",
            color_discrete_map=CLUSTER_COLORS if 'LISA_cluster_label' in gdf.columns else None,
            labels={'x': 'Longitude', 'y': 'Latitude'}
        )
        fig_scatter.update_layout(height=500)
        
        return fig_bar, fig_hist, fig_box, fig_scatter

def create_file_uploader():
    """Cria interface para upload de arquivos"""
    st.markdown("""
    <div class="warning-box">
        <h3>📁 Upload de Arquivos Necessário</h3>
        <p>Os arquivos de dados não foram encontrados. Faça upload dos arquivos:</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dados de Abandono")
        uploaded_abandono = st.file_uploader(
            "📊 Upload: txabandono-municipios.xlsx", 
            type=['xlsx', 'xls'],
            help="Arquivo Excel com dados de abandono escolar por município e ano"
        )
        
        if uploaded_abandono:
            st.success("✅ Arquivo de abandono carregado!")
    
    with col2:
        st.subheader("Dados Geográficos")
        uploaded_geo = st.file_uploader(
            "🗺️ Upload: municipios.csv", 
            type=['csv'],
            help="Arquivo CSV com coordenadas geográficas dos municípios"
        )
        
        if uploaded_geo:
            st.success("✅ Arquivo geográfico carregado!")
    
    if uploaded_abandono and uploaded_geo:
        try:
            with st.spinner("Processando arquivos..."):
                df = pd.read_excel(uploaded_abandono)
                df_geo = pd.read_csv(uploaded_geo, encoding="latin1")
                
                df = DataLoader.process_abandono_data(df)
                
                st.markdown("""
                <div class="success-box">
                    <h4>✅ Arquivos processados com sucesso!</h4>
                </div>
                """, unsafe_allow_html=True)
                
                return df, df_geo, True
        except Exception as e:
            st.error(f"❌ Erro ao processar arquivos: {e}")
            return None, None, False
    
    return None, None, False

def display_geospatial_warning():
    """Exibe aviso sobre funcionalidades geoespaciais"""
    if not GEOSPATIAL_AVAILABLE:
        st.markdown("""
        <div class="warning-box">
            <h3>⚠️ Funcionalidades Limitadas</h3>
            <p>Bibliotecas geoespaciais não disponíveis. Funcionalidades limitadas:</p>
            <ul>
                <li>❌ Cálculo LISA não disponível</li>
                <li>❌ Análise de autocorrelação espacial limitada</li>
                <li>✅ Análise estatística disponível</li>
                <li>✅ Visualizações geográficas básicas disponíveis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def display_results(gdf, ano: int):
    """Exibe os resultados da análise - VERSÃO CORRIGIDA"""
    # Métricas principais
    st.subheader(f"📈 Resultados da Análise - {ano}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Municípios", f"{len(gdf):,}")
    
    with col2:
        if 'LISA_p' in gdf.columns:
            significativos = len(gdf[gdf['LISA_p'] < 0.05])
            percentual = (significativos / len(gdf)) * 100
            st.metric("Clusters Significativos", f"{significativos:,} ({percentual:.1f}%)")
        else:
            st.metric("Taxa Média", f"{gdf['taxa_abandono'].mean():.2f}%")
    
    with col3:
        st.metric("Taxa Média", f"{gdf['taxa_abandono'].mean():.2f}%")
    
    with col4:
        if 'LISA_I' in gdf.columns:
            st.metric("Índice de Moran (médio)", f"{gdf['LISA_I'].mean():.3f}")
        else:
            st.metric("Desvio Padrão", f"{gdf['taxa_abandono'].std():.2f}%")
    
    # Tabs para visualizações
    tabs_list = ["📊 Gráficos", "🗺️ Mapas", "📋 Dados"]
    if 'LISA_cluster_label' in gdf.columns:
        tabs_list.insert(0, "🎯 Clusters LISA")
    
    tabs = st.tabs(tabs_list)
    tab_index = 0
    
    # Tab Clusters LISA (se disponível)
    if 'LISA_cluster_label' in gdf.columns:
        with tabs[tab_index]:
            display_lisa_analysis(gdf)
        tab_index += 1
    
    # Tab Gráficos
    with tabs[tab_index]:
        display_charts(gdf)
    tab_index += 1
    
    # Tab Mapas - VERSÃO CORRIGIDA
    with tabs[tab_index]:
        display_maps_corrected(gdf, ano)
    tab_index += 1
    
    # Tab Dados
    with tabs[tab_index]:
        display_data_table(gdf, ano)

def display_lisa_analysis(gdf):
    """Exibe análise LISA"""
    st.subheader("Análise de Clusters LISA")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Distribuição de Clusters:**")
        cluster_summary = gdf['LISA_cluster_label'].value_counts()
        for cluster, count in cluster_summary.items():
            percentage = (count / len(gdf)) * 100
            desc = CLUSTER_DESCRIPTIONS.get(cluster, cluster)
            st.write(f"• **{cluster}**: {count} municípios ({percentage:.1f}%)")
            st.caption(desc)
    
    with col2:
        # Gráfico de pizza dos clusters
        fig_pie = px.pie(
            values=cluster_summary.values,
            names=cluster_summary.index,
            title="Proporção de Clusters",
            color=cluster_summary.index,
            color_discrete_map=CLUSTER_COLORS
        )
        st.plotly_chart(fig_pie, use_container_width=True)

def display_charts(gdf):
    """Exibe gráficos estatísticos"""
    st.subheader("Análise Estatística")
    
    fig_bar, fig_hist, fig_box, fig_scatter = ChartCreator.create_plotly_charts(gdf)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_bar, use_container_width=True)
    with col2:
        st.plotly_chart(fig_hist, use_container_width=True)
    
    if fig_box:
        st.plotly_chart(fig_box, use_container_width=True)
    
    st.plotly_chart(fig_scatter, use_container_width=True)

def display_maps_corrected(gdf, ano: int):
    """Exibe mapas - VERSÃO CORRIGIDA SEM ERRO DE SERIALIZAÇÃO"""
    st.subheader("Visualização Espacial")
    
    if FOLIUM_AVAILABLE:
        try:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Mapa de Clusters**")
                mapa_cluster = MapVisualizer.create_simple_folium_map(gdf, ano, "cluster")
                if mapa_cluster is not None:
                    # Usar st_folium com parâmetros seguros
                    map_data = st_folium(
                        mapa_cluster, 
                        width=350, 
                        height=400,
                        returned_objects=["last_object_clicked"]  # Limitar objetos retornados
                    )
                else:
                    st.error("Erro ao criar mapa de clusters")
            
            with col2:
                st.write("**Mapa de Significância**")
                mapa_sig = MapVisualizer.create_simple_folium_map(gdf, ano, "significance")
                if mapa_sig is not None:
                    # Usar st_folium com parâmetros seguros
                    map_data2 = st_folium(
                        mapa_sig, 
                        width=350, 
                        height=400,
                        returned_objects=["last_object_clicked"]  # Limitar objetos retornados
                    )
                else:
                    st.error("Erro ao criar mapa de significância")
        
        except Exception as e:
            st.error(f"Erro ao exibir mapas: {e}")
            # Fallback para scatter plot
            display_scatter_fallback(gdf)
    else:
        st.warning("Folium não disponível - usando visualização alternativa")
        display_scatter_fallback(gdf)

def display_scatter_fallback(gdf):
    """Exibe scatter plot como fallback para mapas"""
    st.subheader("Distribuição Geográfica (Scatter Plot)")
    
    if 'longitude' in gdf.columns and 'latitude' in gdf.columns:
        fig_scatter = px.scatter(
            gdf,
            x='longitude',
            y='latitude',
            color='LISA_cluster_label' if 'LISA_cluster_label' in gdf.columns else 'taxa_abandono',
            size='taxa_abandono',
            hover_name='municipio' if 'municipio' in gdf.columns else None,
            title="Distribuição Espacial dos Municípios",
            color_discrete_map=CLUSTER_COLORS if 'LISA_cluster_label' in gdf.columns else None
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.warning("Coordenadas geográficas não disponíveis")

def display_data_table(gdf, ano: int):
    """Exibe tabela de dados"""
    st.subheader("Dados Detalhados")
    
    # Filtros
    col1, col2 = st.columns(2)
    
    with col1:
        if 'LISA_cluster_label' in gdf.columns:
            cluster_filter = st.multiselect(
                "Filtrar por cluster:",
                options=gdf['LISA_cluster_label'].unique(),
                default=gdf['LISA_cluster_label'].unique()
            )
            gdf_filtered = gdf[gdf['LISA_cluster_label'].isin(cluster_filter)]
        else:
            gdf_filtered = gdf
    
    with col2:
        if 'UF' in gdf.columns:
            uf_filter = st.multiselect(
                "Filtrar por UF:",
                options=sorted(gdf['UF'].dropna().unique()),
                default=sorted(gdf['UF'].dropna().unique())
            )
            gdf_filtered = gdf_filtered[gdf_filtered['UF'].isin(uf_filter)]
    
    # Colunas para mostrar
    colunas_mostrar = ['municipio', 'UF', 'Região', 'taxa_abandono']
    if 'LISA_cluster_label' in gdf.columns:
        colunas_mostrar.extend(['LISA_cluster_label', 'LISA_I', 'LISA_p'])
    
    colunas_disponiveis = [col for col in colunas_mostrar if col in gdf_filtered.columns]
    
    # Mostrar dados
    st.dataframe(gdf_filtered[colunas_disponiveis].round(3), use_container_width=True)
    
    # Download
    csv_data = gdf_filtered[colunas_disponiveis].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Baixar dados filtrados (CSV)",
        data=csv_data,
        file_name=f"lisa_dados_{ano}.csv",
        mime="text/csv"
    )

def main():
    """Função principal da aplicação"""
    load_css()
    
    # Título principal
    st.markdown('<h1 class="main-header">📊 Análise Espacial LISA - Abandono Escolar</h1>', unsafe_allow_html=True)
    
    # Banner informativo
    st.markdown("""
    <div class="data-info">
        <h3>🎯 Análise de Autocorrelação Espacial</h3>
        <p>Esta aplicação analisa dados de abandono escolar no ensino médio brasileiro usando 
        técnicas de análise espacial LISA (Local Indicators of Spatial Association).</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Verificar disponibilidade de bibliotecas
    display_geospatial_warning()
    
    # Carregar dados
    with st.spinner("🔄 Carregando dados do sistema..."):
        df, df_geo, data_loaded = DataLoader.load_data()
    
    # Se não conseguir carregar, oferecer upload
    if not data_loaded:
        df, df_geo, data_loaded = create_file_uploader()
    
    if data_loaded and df is not None and df_geo is not None:
        # Sidebar com informações
        st.sidebar.header("📊 Informações dos Dados")
        st.sidebar.success(f"""
        **✅ Dados Carregados!**
        
        **Municípios:** {len(df['cod_mun'].unique()):,}  
        **Anos:** {len(df['Ano'].unique())}  
        **Registros:** {len(df):,}  
        **Período:** {df['Ano'].min()} - {df['Ano'].max()}
        """)
        
        # Anos disponíveis
        anos_disponiveis = sorted(df['Ano'].unique())
        
        # Seleção de ano
        st.sidebar.header("🎯 Configurações")
        ano_selecionado = st.sidebar.selectbox(
            "📅 Ano para Análise",
            anos_disponiveis,
            index=len(anos_disponiveis)-1,
            help="Selecione o ano que deseja analisar"
        )
        
        # Calcular LISA (se disponível)
        gdf = None
        if GEOSPATIAL_AVAILABLE:
            with st.spinner(f"Calculando estatísticas LISA para {ano_selecionado}..."):
                gdf = LISACalculator.calculate_lisa_for_year(df, df_geo, ano_selecionado)
        else:
            # Criar dados básicos sem LISA
            df_ano = df[df["Ano"] == ano_selecionado].copy()
            if not df_ano.empty:
                df_media = df_ano.groupby("cod_mun")["taxa"].mean().reset_index()
                df_media = df_media.rename(columns={"taxa": "taxa_abandono"})
                
                # Preparar dados geográficos
                df_geo_clean = df_geo.copy()
                if 'cod_mun' not in df_geo_clean.columns:
                    df_geo_clean = df_geo_clean.rename(columns={"code_muni": "cod_mun"})
                df_geo_clean["cod_mun"] = df_geo_clean["cod_mun"].astype(int)
                
                gdf = df_geo_clean.merge(df_media, on="cod_mun", how="inner")
                gdf = gdf.merge(df_ano[["cod_mun", "UF", "Região"]].drop_duplicates(),
                               on="cod_mun", how="left")
        
        if gdf is not None and len(gdf)