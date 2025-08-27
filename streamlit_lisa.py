import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from pathlib import Path

# Configuração da página
st.set_page_config(
    page_title="Análise Espacial LISA - Deploy Safe", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
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

# Título principal
st.markdown('<h1 class="main-header">📊 Análise Espacial LISA - Versão Deploy</h1>', unsafe_allow_html=True)

# Banner informativo
st.markdown("""
<div class="data-info">
    <h3>🚀 Versão Otimizada para Deploy</h3>
    <p>Esta versão foi otimizada para funcionar em plataformas de deploy como Streamlit Cloud, 
    com dependências mínimas e máxima compatibilidade.</p>
</div>
""", unsafe_allow_html=True)

# Função para tentar importar GeoPandas
@st.cache_data
def try_import_geopandas():
    """Tenta importar GeoPandas, retorna False se falhar"""
    try:
        import geopandas as gpd
        from libpysal.weights import Queen
        from esda.moran import Moran_Local
        return True, gpd, Queen, Moran_Local
    except ImportError as e:
        st.error(f"⚠️ Dependências espaciais não disponíveis: {e}")
        st.info("💡 Executando em modo simplificado sem análise LISA")
        return False, None, None, None

# Verificar disponibilidade de dependências espaciais
SPATIAL_AVAILABLE, gpd, Queen, Moran_Local = try_import_geopandas()

# Cache para carregamento dos dados
@st.cache_data
def load_sample_data():
    """Carrega dados de exemplo se os arquivos reais não estiverem disponíveis"""
    try:
        # Tentar carregar dados reais primeiro
        base_path = Path(".")
        abandono_path = base_path / "data" / "txabandono-municipios.xlsx"
        municipios_path = base_path / "data" / "municipios.csv"
        
        if abandono_path.exists() and municipios_path.exists():
            # Carregar dados reais
            df = pd.read_excel(abandono_path)
            df_geo = pd.read_csv(municipios_path, encoding="latin1")
            
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
            
            return df, df_geo, True, "real"
        else:
            # Gerar dados de exemplo
            return generate_sample_data()
            
    except Exception as e:
        st.warning(f"⚠️ Erro ao carregar dados reais: {e}")
        return generate_sample_data()

def generate_sample_data():
    """Gera dados de exemplo para demonstração"""
    np.random.seed(42)
    
    # Gerar dados de exemplo
    n_municipios = 100
    anos = [2020, 2021, 2022, 2023]
    
    # Dados de abandono
    data_abandono = []
    for ano in anos:
        for i in range(n_municipios):
            data_abandono.append({
                'cod_mun': i + 1,
                'Ano': ano,
                'taxa': np.random.normal(5, 2),  # Taxa média de 5% com desvio de 2%
                'UF': np.random.choice(['SP', 'RJ', 'MG', 'RS', 'PR']),
                'Região': np.random.choice(['Sudeste', 'Sul', 'Nordeste'])
            })
    
    df_abandono = pd.DataFrame(data_abandono)
    df_abandono['taxa'] = np.clip(df_abandono['taxa'], 0, 20)  # Limitar entre 0 e 20%
    
    # Dados geográficos
    data_geo = []
    for i in range(n_municipios):
        data_geo.append({
            'cod_mun': i + 1,
            'municipio': f'Município {i+1}',
            'latitude': np.random.uniform(-33, 5),  # Latitude do Brasil
            'longitude': np.random.uniform(-74, -34)  # Longitude do Brasil
        })
    
    df_geo = pd.DataFrame(data_geo)
    
    return df_abandono, df_geo, True, "sample"

@st.cache_data
def calculate_simple_statistics(df, df_geo, ano):
    """Calcula estatísticas simples sem LISA (fallback)"""
    # Filtrar dados do ano
    df_ano = df[df['Ano'] == ano].copy()
    if df_ano.empty:
        return None
    
    # Calcular média por município
    df_media = (
        df_ano.groupby('cod_mun')['taxa']
        .mean()
        .reset_index()
        .rename(columns={'taxa': 'taxa_abandono'})
    )
    
    # Juntar com dados geográficos
    df_combined = df_geo.merge(df_media, on='cod_mun', how='inner')
    
    # Adicionar classificação simples baseada em quartis
    quartis = df_combined['taxa_abandono'].quantile([0.25, 0.5, 0.75])
    
    def classify_rate(rate):
        if rate <= quartis[0.25]:
            return "Baixo"
        elif rate <= quartis[0.5]:
            return "Médio-Baixo"
        elif rate <= quartis[0.75]:
            return "Médio-Alto"
        else:
            return "Alto"
    
    df_combined['classificacao'] = df_combined['taxa_abandono'].apply(classify_rate)
    
    # Juntar com informações regionais
    df_combined = df_combined.merge(
        df_ano[['cod_mun', 'UF', 'Região']].drop_duplicates(),
        on='cod_mun', 
        how='left'
    )
    
    return df_combined

@st.cache_data
def calculate_lisa_analysis(df, df_geo, ano):
    """Calcula análise LISA completa (se disponível)"""
    if not SPATIAL_AVAILABLE:
        return calculate_simple_statistics(df, df_geo, ano)
    
    try:
        # Preparar dados
        df_geo_renamed = df_geo.rename(columns={"cod_mun": "code_muni"})
        df_geo_renamed["code_muni"] = df_geo_renamed["code_muni"].astype(int)
        
        # Criar GeoDataFrame
        gdf_base = gpd.GeoDataFrame(
            df_geo_renamed,
            geometry=gpd.points_from_xy(df_geo_renamed["longitude"], df_geo_renamed["latitude"]),
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

        # Juntar informações regionais
        gdf = gdf.merge(df_ano[["cod_mun", "UF", "Região"]].drop_duplicates(),
                        left_on="code_muni", right_on="cod_mun", how="left")

        return gdf
        
    except Exception as e:
        st.warning(f"⚠️ Erro na análise LISA: {e}")
        return calculate_simple_statistics(df, df_geo, ano)

def create_simple_map(df, ano, map_type="classification"):
    """Cria mapa simples com Folium"""
    # Cores para classificação
    cores_classificacao = {
        "Baixo": "#1f77b4",     # Azul
        "Médio-Baixo": "#2ca02c", # Verde
        "Médio-Alto": "#ff7f0e",  # Laranja
        "Alto": "#d62728"         # Vermelho
    }
    
    # Cores para clusters LISA (se disponível)
    cores_cluster = {
        "HH": "#d62728",  # Vermelho
        "LL": "#1f77b4",  # Azul
        "LH": "#2ca02c",  # Verde
        "HL": "#ff7f0e",  # Laranja
        "ns": "#7f7f7f"   # Cinza
    }
    
    # Criar mapa base centrado no Brasil
    m = folium.Map(
        location=[-14.2350, -51.9253],
        zoom_start=4,
        tiles='OpenStreetMap'
    )
    
    # Adicionar pontos ao mapa
    for idx, row in df.iterrows():
        if map_type == "classification" and 'classificacao' in df.columns:
            color = cores_classificacao.get(row["classificacao"], "#7f7f7f")
            popup_text = f"""
            <b>{row.get('municipio', 'N/A')}</b><br>
            UF: {row.get('UF', 'N/A')}<br>
            Região: {row.get('Região', 'N/A')}<br>
            Taxa Abandono: {row['taxa_abandono']:.2f}%<br>
            Classificação: {row['classificacao']}
            """
        elif map_type == "cluster" and 'LISA_cluster_label' in df.columns:
            color = cores_cluster.get(row["LISA_cluster_label"], "#7f7f7f")
            popup_text = f"""
            <b>{row.get('municipio', 'N/A')}</b><br>
            UF: {row.get('UF', 'N/A')}<br>
            Região: {row.get('Região', 'N/A')}<br>
            Taxa Abandono: {row['taxa_abandono']:.2f}%<br>
            Cluster: {row['LISA_cluster_label']}<br>
            p-valor: {row.get('LISA_p', 'N/A')}
            """
        else:
            color = "#1f77b4"
            popup_text = f"""
            <b>{row.get('municipio', 'N/A')}</b><br>
            Taxa Abandono: {row['taxa_abandono']:.2f}%
            """
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=4,
            popup=popup_text,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7
        ).add_to(m)
    
    return m

def create_charts(df):
    """Cria gráficos com Plotly"""
    charts = []
    
    # Gráfico 1: Distribuição das taxas
    fig_hist = px.histogram(
        df, 
        x='taxa_abandono',
        title="Distribuição das Taxas de Abandono",
        labels={'taxa_abandono': 'Taxa de Abandono (%)', 'count': 'Frequência'},
        nbins=20
    )
    charts.append(("Distribuição", fig_hist))
    
    # Gráfico 2: Por classificação ou cluster
    if 'classificacao' in df.columns:
        class_counts = df['classificacao'].value_counts()
        fig_bar = px.bar(
            x=class_counts.index,
            y=class_counts.values,
            title="Distribuição por Classificação",
            labels={'x': 'Classificação', 'y': 'Número de Municípios'},
            color=class_counts.index
        )
        charts.append(("Classificação", fig_bar))
    
    if 'LISA_cluster_label' in df.columns:
        cluster_counts = df['LISA_cluster_label'].value_counts()
        fig_cluster = px.bar(
            x=cluster_counts.index,
            y=cluster_counts.values,
            title="Distribuição de Clusters LISA",
            labels={'x': 'Tipo de Cluster', 'y': 'Número de Municípios'},
            color=cluster_counts.index
        )
        charts.append(("Clusters LISA", fig_cluster))
    
    # Gráfico 3: Por região (se disponível)
    if 'Região' in df.columns:
        fig_box = px.box(
            df, 
            x='Região', 
            y='taxa_abandono',
            title="Taxa de Abandono por Região",
            labels={'taxa_abandono': 'Taxa de Abandono (%)', 'Região': 'Região'}
        )
        charts.append(("Por Região", fig_box))
    
    return charts

# Interface principal
def main():
    # Carregar dados
    with st.spinner("🔄 Carregando dados..."):
        df, df_geo, data_loaded, data_type = load_sample_data()

    if data_loaded:
        # Informações sobre os dados
        st.sidebar.header("📊 Informações dos Dados")
        
        if data_type == "sample":
            st.sidebar.warning("⚠️ Usando dados de exemplo")
            st.sidebar.info("Para usar dados reais, coloque os arquivos na pasta data/")
        else:
            st.sidebar.success("✅ Dados reais carregados")
        
        st.sidebar.info(f"""
        **Municípios:** {len(df['cod_mun'].unique())}  
        **Anos disponíveis:** {len(df['Ano'].unique())}  
        **Total de registros:** {len(df)}
        """)
        
        # Seleção de ano
        anos_disponiveis = sorted(df['Ano'].unique())
        ano_selecionado = st.sidebar.selectbox(
            "📅 Selecione o Ano para Análise",
            anos_disponiveis,
            index=len(anos_disponiveis)-1
        )
        
        # Opções de visualização
        st.sidebar.header("🔍 Opções de Visualização")
        mostrar_mapas = st.sidebar.checkbox("🗺️ Mostrar mapas", value=True)
        mostrar_graficos = st.sidebar.checkbox("📈 Mostrar gráficos", value=True)
        mostrar_dados = st.sidebar.checkbox("📋 Mostrar dados", value=True)
        
        # Calcular análise para o ano selecionado
        with st.spinner(f"🧮 Processando dados para {ano_selecionado}..."):
            if SPATIAL_AVAILABLE:
                resultado = calculate_lisa_analysis(df, df_geo, ano_selecionado)
                analysis_type = "LISA"
            else:
                resultado = calculate_simple_statistics(df, df_geo, ano_selecionado)
                analysis_type = "Estatística Simples"
        
        if resultado is not None:
            # Métricas principais
            st.subheader(f"📊 Resultados da Análise {analysis_type} - {ano_selecionado}")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🏘️ Total de Municípios", len(resultado))
            
            with col2:
                if 'LISA_p' in resultado.columns:
                    significativos = len(resultado[resultado['LISA_p'] < 0.05])
                    st.metric("✅ Clusters Significativos", significativos)
                else:
                    altos = len(resultado[resultado['classificacao'] == 'Alto'])
                    st.metric("🔴 Taxa Alta", altos)
            
            with col3:
                taxa_media = resultado['taxa_abandono'].mean()
                st.metric("📈 Taxa Média", f"{taxa_media:.2f}%")
            
            with col4:
                if 'LISA_I' in resultado.columns:
                    moran_i = resultado['LISA_I'].mean()
                    st.metric("🔗 Moran's I", f"{moran_i:.3f}")
                else:
                    desvio = resultado['taxa_abandono'].std()
                    st.metric("📊 Desvio Padrão", f"{desvio:.2f}%")
            
            # Visualizações
            if mostrar_mapas:
                st.subheader("🗺️ Mapas Interativos")
                
                if 'LISA_cluster_label' in resultado.columns:
                    # Mapa LISA
                    st.markdown("**Mapa de Clusters LISA:**")
                    mapa_lisa = create_simple_map(resultado, ano_selecionado, "cluster")
                    st_folium(mapa_lisa, width=700, height=400)
                    
                    st.markdown("""
                    **Legenda:**
                    - 🔴 **HH:** Alta taxa, vizinhos com alta taxa
                    - 🔵 **LL:** Baixa taxa, vizinhos com baixa taxa  
                    - 🟢 **LH:** Baixa taxa, vizinhos com alta taxa
                    - 🟠 **HL:** Alta taxa, vizinhos com baixa taxa
                    - ⚫ **ns:** Não significativo
                    """)
                else:
                    # Mapa de classificação simples
                    st.markdown("**Mapa de Classificação por Quartis:**")
                    mapa_class = create_simple_map(resultado, ano_selecionado, "classification")
                    st_folium(mapa_class, width=700, height=400)
                    
                    st.markdown("""
                    **Legenda:**
                    - 🔵 **Baixo:** 25% menores taxas
                    - 🟢 **Médio-Baixo:** 25%-50%
                    - 🟠 **Médio-Alto:** 50%-75%
                    - 🔴 **Alto:** 25% maiores taxas
                    """)
            
            if mostrar_graficos:
                st.subheader("📈 Análise Estatística")
                
                charts = create_charts(resultado)
                
                # Mostrar gráficos em colunas
                if len(charts) >= 2:
                    col1, col2 = st.columns(2)
                    for i, (title, fig) in enumerate(charts):
                        if i % 2 == 0:
                            with col1:
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            with col2:
                                st.plotly_chart(fig, use_container_width=True)
                else:
                    for title, fig in charts:
                        st.plotly_chart(fig, use_container_width=True)
            
            if mostrar_dados:
                st.subheader("📋 Dados Detalhados")
                
                # Filtros
                if 'LISA_cluster_label' in resultado.columns:
                    cluster_filter = st.multiselect(
                        "Filtrar por cluster:",
                        options=resultado['LISA_cluster_label'].unique(),
                        default=resultado['LISA_cluster_label'].unique()
                    )
                    resultado_filtrado = resultado[resultado['LISA_cluster_label'].isin(cluster_filter)]
                elif 'classificacao' in resultado.columns:
                    class_filter = st.multiselect(
                        "Filtrar por classificação:",
                        options=resultado['classificacao'].unique(),
                        default=resultado['classificacao'].unique()
                    )
                    resultado_filtrado = resultado[resultado['classificacao'].isin(class_filter)]
                else:
                    resultado_filtrado = resultado
                
                # Mostrar tabela
                colunas_mostrar = ['municipio', 'UF', 'Região', 'taxa_abandono']
                if 'LISA_cluster_label' in resultado_filtrado.columns:
                    colunas_mostrar.extend(['LISA_cluster_label', 'LISA_I', 'LISA_p'])
                elif 'classificacao' in resultado_filtrado.columns:
                    colunas_mostrar.append('classificacao')
                
                colunas_disponiveis = [col for col in colunas_mostrar if col in resultado_filtrado.columns]
                
                st.dataframe(
                    resultado_filtrado[colunas_disponiveis].round(3),
                    use_container_width=True
                )
                
                # Download
                csv_data = resultado_filtrado[colunas_disponiveis].to_csv(index=False)
                st.download_button(
                    "📥 Baixar dados (CSV)",
                    csv_data,
                    f"dados_analise_{ano_selecionado}.csv",
                    "text/csv"
                )
            
            # Resumo estatístico
            st.subheader("📊 Resumo Estatístico")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'LISA_cluster_label' in resultado.columns:
                    st.write("**Distribuição de Clusters LISA:**")
                    cluster_summary = resultado['LISA_cluster_label'].value_counts()
                    for cluster, count in cluster_summary.items():
                        percentage = (count / len(resultado)) * 100
                        st.write(f"- **{cluster}**: {count} ({percentage:.1f}%)")
                elif 'classificacao' in resultado.columns:
                    st.write("**Distribuição por Classificação:**")
                    class_summary = resultado['classificacao'].value_counts()
                    for classe, count in class_summary.items():
                        percentage = (count / len(resultado)) * 100
                        st.write(f"- **{classe}**: {count} ({percentage:.1f}%)")
            
            with col2:
                st.write("**Estatísticas da Taxa de Abandono:**")
                st.write(f"- **Média**: {resultado['taxa_abandono'].mean():.2f}%")
                st.write(f"- **Mediana**: {resultado['taxa_abandono'].median():.2f}%")
                st.write(f"- **Desvio padrão**: {resultado['taxa_abandono'].std():.2f}%")
                st.write(f"- **Mínimo**: {resultado['taxa_abandono'].min():.2f}%")
                st.write(f"- **Máximo**: {resultado['taxa_abandono'].max():.2f}%")
        
        else:
            st.error(f"❌ Não foi possível processar os dados para o ano {ano_selecionado}")
    
    else:
        st.error("❌ Erro ao carregar dados")

# Informações sobre dependências
if not SPATIAL_AVAILABLE:
    st.sidebar.warning("""
    ⚠️ **Modo Simplificado**
    
    Algumas dependências espaciais não estão disponíveis.
    O aplicativo está executando com funcionalidades básicas.
    
    Para análise LISA completa, instale:
    - geopandas
    - libpysal
    - esda
    """)

# Executar aplicativo
if __name__ == "__main__":
    main()

