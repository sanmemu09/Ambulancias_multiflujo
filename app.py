# -*- coding: utf-8 -*-
"""
🚑 Ambulance Multi-Flow Optimization System
Streamlit Application for Optimal Ambulance Dispatch
"""

import streamlit as st
import osmnx as ox
import networkx as nx
import pandas as pd
import random
from pulp import *
import folium
from streamlit_folium import st_folium
from collections import Counter
import time

# ============================================================================
# CONFIGURATION - IMMUTABLE PARAMETERS
# ============================================================================

# Geographic parameters (FIXED - from original model)
LATITUD_CENTRO = 6.2442
LONGITUD_CENTRO = -75.5812
RADIO_METROS = 500

# Cost coefficients (FIXED)
COEF_PERSONAL = 10
COEF_EQUIPAMIENTO = 5
COEF_INSUMOS = 3

# Objective function weights (FIXED)
BETA = 0.5    # Time weight
GAMMA = 1.0   # Operational cost weight

# Big-M constraint
M = 10000

# Page configuration
st.set_page_config(
    page_title="Ambulance Optimizer",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# AMBULANCE CLASS (from original code)
# ============================================================================

class Ambulancia:
    """Represents an ambulance with resources and operational cost."""
    
    def __init__(self, id, personal, equipamiento, insumos):
        self.id = id
        self.personal = personal
        self.equipamiento = equipamiento
        self.insumos = insumos
        self.costo_operativo = self._calcular_costo_operativo()
        self.tipo = self._asignar_tipo_ambulancia()
    
    def _calcular_costo_operativo(self):
        return (self.personal * COEF_PERSONAL + 
                self.equipamiento * COEF_EQUIPAMIENTO + 
                self.insumos * COEF_INSUMOS)
    
    def _asignar_tipo_ambulancia(self):
        if self.costo_operativo >= 200:
            return 'Crítica'
        elif self.costo_operativo >= 100:
            return 'Media'
        elif self.costo_operativo >= 50:
            return 'Leve'
        else:
            return 'Desconocido'
    
    def to_dict(self):
        return {
            'ID': self.id,
            'Personal': self.personal,
            'Equipamiento': self.equipamiento,
            'Insumos': self.insumos,
            'Costo Operativo': self.costo_operativo,
            'Tipo': self.tipo
        }

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def es_compatible(ambulancia, tipo_emergencia):
    """Check if ambulance can handle emergency type."""
    niveles_ambulancia = {'Crítica': 3, 'Media': 2, 'Leve': 1}
    niveles_emergencia = {'Crítica': 3, 'Media': 2, 'Leve': 1}
    
    nivel_amb = niveles_ambulancia.get(ambulancia.tipo, 0)
    nivel_emerg = niveles_emergencia.get(tipo_emergencia, 0)
    
    return nivel_amb >= nivel_emerg

@st.cache_data(show_spinner=False)
def descargar_red(lat, lon, radio):
    """Download street network from OSM."""
    try:
        G = ox.graph_from_point(
            (lat, lon),
            dist=radio,
            network_type='drive'
        )
        return G
    except Exception as e:
        st.error(f"Error downloading network: {str(e)}")
        return None

def asignar_capacidades_velocidades(G, C_MIN, C_MAX, R_MIN, R_MAX):
    """Assign capacities to edges and calculate travel times."""
    arcos = list(G.edges(keys=True))
    
    # Assign capacities
    for u, v, key in arcos:
        capacidad = random.uniform(C_MIN, C_MAX)
        G.edges[(u, v, key)]['Capacidad_C'] = capacidad
        
        # Calculate distance
        if 'length' in G.edges[(u, v, key)]:
            distancia_km = G.edges[(u, v, key)]['length'] / 1000
        else:
            x1, y1 = G.nodes[u]['x'], G.nodes[u]['y']
            x2, y2 = G.nodes[v]['x'], G.nodes[v]['y']
            distancia_km = ((x2-x1)**2 + (y2-y1)**2)**0.5 * 111
        
        G.edges[(u, v, key)]['Distancia'] = distancia_km
        tiempo_horas = distancia_km / capacidad
        G.edges[(u, v, key)]['tiempo_min'] = tiempo_horas * 60
    
    # Assign required speeds by emergency type
    rango = R_MAX - R_MIN
    tercio = rango / 3
    
    R_k = {
        'Leve': random.uniform(R_MIN, R_MIN + tercio),
        'Media': random.uniform(R_MIN + tercio, R_MIN + 2*tercio),
        'Crítica': random.uniform(R_MIN + 2*tercio, R_MAX)
    }
    
    return R_k

def generar_origen_destinos(G, num_incidentes):
    """Generate origin and connected destinations."""
    # Find largest strongly connected component
    if not nx.is_strongly_connected(G):
        componentes = list(nx.strongly_connected_components(G))
        componente_mas_grande = max(componentes, key=len)
        componente_principal = G.subgraph(componente_mas_grande).copy()
    else:
        componente_principal = G
    
    nodos_componente = list(componente_principal.nodes())
    
    # Select origin (most central node)
    try:
        centralidad = nx.betweenness_centrality(componente_principal)
        ORIGEN = max(centralidad, key=centralidad.get)
    except:
        grados = {n: componente_principal.degree(n) for n in nodos_componente}
        ORIGEN = max(grados, key=grados.get)
    
    # Select reachable destinations
    nodos_alcanzables = list(nx.descendants(componente_principal, ORIGEN))
    nodos_alcanzables.append(ORIGEN)
    
    candidatos = [n for n in nodos_alcanzables 
                  if n != ORIGEN and componente_principal.degree(n) > 0]
    
    if len(candidatos) < num_incidentes:
        num_incidentes = len(candidatos)
    
    destinos_nodos = random.sample(candidatos, num_incidentes)
    
    # Assign emergency types
    FLUJOS = ['Crítica', 'Media', 'Leve']
    tipos_emergencia = (FLUJOS * (num_incidentes // len(FLUJOS) + 1))[:num_incidentes]
    random.shuffle(tipos_emergencia)
    
    DESTINOS = {nodo: tipo for nodo, tipo in zip(destinos_nodos, tipos_emergencia)}
    
    return ORIGEN, DESTINOS

def resolver_optimizacion(G, ambulancias, ORIGEN, DESTINOS, R_k, factor_relajacion):
    """Solve the multi-flow optimization model."""
    
    nodos = list(G.nodes())
    arcos = list(G.edges(keys=True))
    
    # Create priority mapping
    prioridades = {'Crítica': 3, 'Media': 2, 'Leve': 1}
    
    # Create incidents list
    incidentes = []
    for nodo_dest, tipo_emerg in DESTINOS.items():
        incidentes.append({
            'nodo': nodo_dest,
            'tipo': tipo_emerg,
            'prioridad': prioridades[tipo_emerg],
            'velocidad_requerida': R_k[tipo_emerg]
        })
    
    # Create ambulance dictionary
    ambulancias_dict = {amb.id: amb for amb in ambulancias}
    
    # Create compatible pairs
    pares_amb_inc = []
    for amb in ambulancias:
        for inc in incidentes:
            if es_compatible(amb, inc['tipo']):
                pares_amb_inc.append((amb.id, inc['nodo']))
    
    if not pares_amb_inc:
        return None, "No compatible ambulance-incident pairs found"
    
    # Create optimization model
    modelo = LpProblem("Despacho_Ambulancias", LpMinimize)
    
    # Decision variables
    y = {}
    for amb_id, nodo_dest in pares_amb_inc:
        y[(amb_id, nodo_dest)] = LpVariable(f"y_{amb_id}_{nodo_dest}", cat='Binary')
    
    x = {}
    for amb_id, nodo_dest in pares_amb_inc:
        for u, v, key in arcos:
            x[(amb_id, nodo_dest, u, v, key)] = LpVariable(
                f"x_{amb_id}_{nodo_dest}_{u}_{v}_{key}", cat='Binary'
            )
    
    T = {}
    for amb_id, nodo_dest in pares_amb_inc:
        T[(amb_id, nodo_dest)] = LpVariable(
            f"T_{amb_id}_{nodo_dest}", lowBound=0, cat='Continuous'
        )
    
    # Constraint 1: Each incident attended by exactly one ambulance
    for inc in incidentes:
        nodo_dest = inc['nodo']
        ambulancias_compatibles = [
            (amb_id, nodo_dest) for amb_id, nd in pares_amb_inc if nd == nodo_dest
        ]
        modelo += (
            lpSum([y[par] for par in ambulancias_compatibles]) == 1,
            f"Incidente_{nodo_dest}_atendido"
        )
    
    # Constraint 2: Each ambulance attends at most one incident
    for amb in ambulancias:
        incidentes_disponibles = [
            (amb.id, nodo_dest) for aid, nodo_dest in pares_amb_inc if aid == amb.id
        ]
        if incidentes_disponibles:
            modelo += (
                lpSum([y[par] for par in incidentes_disponibles]) <= 1,
                f"Ambulancia_{amb.id}_max1"
            )
    
    # Constraint 3: Flow conservation
    for amb_id, nodo_dest in pares_amb_inc:
        for nodo in nodos:
            flujo_salida = lpSum([
                x[(amb_id, nodo_dest, u, v, key)]
                for u, v, key in arcos if u == nodo
            ])
            flujo_entrada = lpSum([
                x[(amb_id, nodo_dest, u, v, key)]
                for u, v, key in arcos if v == nodo
            ])
            
            if nodo == ORIGEN:
                modelo += (
                    flujo_salida - flujo_entrada == y[(amb_id, nodo_dest)],
                    f"Flujo_origen_{amb_id}_{nodo_dest}_{nodo}"
                )
            elif nodo == nodo_dest:
                modelo += (
                    flujo_entrada - flujo_salida == y[(amb_id, nodo_dest)],
                    f"Flujo_destino_{amb_id}_{nodo_dest}_{nodo}"
                )
            else:
                modelo += (
                    flujo_entrada - flujo_salida == 0,
                    f"Flujo_intermedio_{amb_id}_{nodo_dest}_{nodo}"
                )
    
    # Constraint 4: Time calculation
    for amb_id, nodo_dest in pares_amb_inc:
        tipo_emerg = DESTINOS[nodo_dest]
        vel_flujo = R_k[tipo_emerg]
        
        tiempo_total = lpSum([
            x[(amb_id, nodo_dest, u, v, key)] *
            (G.edges[(u, v, key)]['Distancia'] / vel_flujo) * 60
            for u, v, key in arcos
        ])
        
        modelo += (
            T[(amb_id, nodo_dest)] >= tiempo_total,
            f"Tiempo_{amb_id}_{nodo_dest}"
        )
    
    # Constraint 5: Capacity (relaxed)
    for u, v, key in arcos:
        capacidad = G.edges[(u, v, key)]['Capacidad_C']
        capacidad_relajada = capacidad * factor_relajacion
        
        velocidades_en_arco = []
        for amb_id, nodo_dest in pares_amb_inc:
            tipo_emerg = DESTINOS[nodo_dest]
            vel_req = R_k[tipo_emerg]
            velocidades_en_arco.append(
                x[(amb_id, nodo_dest, u, v, key)] * vel_req
            )
        
        modelo += (
            lpSum(velocidades_en_arco) <= capacidad_relajada,
            f"Capacidad_{u}_{v}_{key}"
        )
    
    # Big-M constraints
    for amb_id, nodo_dest in pares_amb_inc:
        modelo += (
            T[(amb_id, nodo_dest)] <= M * y[(amb_id, nodo_dest)],
            f"T_solo_si_asignado_{amb_id}_{nodo_dest}"
        )
    
    # Objective function
    costo_operativo = lpSum([
        prioridades[DESTINOS[nodo_dest]] *
        GAMMA *
        ambulancias_dict[amb_id].costo_operativo *
        x[(amb_id, nodo_dest, u, v, key)] *
        G.edges[(u, v, key)]['Distancia']
        for amb_id, nodo_dest in pares_amb_inc
        for u, v, key in arcos
    ])
    
    costo_tiempo = lpSum([
        prioridades[DESTINOS[nodo_dest]] *
        BETA *
        T[(amb_id, nodo_dest)]
        for amb_id, nodo_dest in pares_amb_inc
    ])
    
    modelo += costo_operativo + costo_tiempo, "Costo_Total"
    
    # Solve
    modelo.solve(PULP_CBC_CMD(msg=0))
    
    if modelo.status != LpStatusOptimal:
        return None, f"Model status: {LpStatus[modelo.status]}"
    
    # Extract results
    asignaciones = []
    rutas_optimas = {}
    
    for amb_id, nodo_dest in pares_amb_inc:
        if y[(amb_id, nodo_dest)].varValue == 1:
            ambulancia = ambulancias_dict[amb_id]
            tipo_emerg = DESTINOS[nodo_dest]
            tiempo = T[(amb_id, nodo_dest)].varValue
            
            # Extract route
            arcos_usados = []
            distancia_total = 0
            
            for u, v, key in arcos:
                if x[(amb_id, nodo_dest, u, v, key)].varValue == 1:
                    arcos_usados.append((u, v, key))
                    distancia_total += G.edges[(u, v, key)]['Distancia']
            
            # Reconstruct path
            ruta = [ORIGEN]
            nodo_actual = ORIGEN
            arcos_restantes = arcos_usados.copy()
            
            while arcos_restantes:
                for arco in arcos_restantes:
                    u, v, key = arco
                    if u == nodo_actual:
                        ruta.append(v)
                        nodo_actual = v
                        arcos_restantes.remove(arco)
                        break
            
            asignaciones.append({
                'ambulancia_id': amb_id,
                'ambulancia_tipo': ambulancia.tipo,
                'costo_operativo': ambulancia.costo_operativo,
                'nodo_destino': nodo_dest,
                'tipo_emergencia': tipo_emerg,
                'prioridad': prioridades[tipo_emerg],
                'tiempo_min': tiempo,
                'velocidad_req': R_k[tipo_emerg],
                'distancia_km': distancia_total
            })
            
            rutas_optimas[amb_id] = {
                'ruta': ruta,
                'arcos': arcos_usados,
                'destino': nodo_dest,
                'distancia_km': distancia_total
            }
    
    resultado = {
        'modelo': modelo,
        'asignaciones': asignaciones,
        'rutas': rutas_optimas,
        'costo_total': value(modelo.objective),
        'origen': ORIGEN,
        'destinos': DESTINOS,
        'R_k': R_k
    }
    
    return resultado, None

def crear_mapa(G, resultado):
    """Create interactive Folium map with results."""
    
    ORIGEN = resultado['origen']
    DESTINOS = resultado['destinos']
    asignaciones = resultado['asignaciones']
    rutas_optimas = resultado['rutas']
    
    lat_origen = G.nodes[ORIGEN]['y']
    lon_origen = G.nodes[ORIGEN]['x']
    
    m = folium.Map(
        location=[lat_origen, lon_origen],
        zoom_start=15,
        tiles='OpenStreetMap'
    )
    
    # Draw street network
    gdf_edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
    for _, edge in gdf_edges.iterrows():
        line = edge['geometry']
        folium.PolyLine(
            locations=[(coord[1], coord[0]) for coord in line.coords],
            color='lightgray',
            weight=2,
            opacity=0.4
        ).add_to(m)
    
    # Colors by emergency type
    colores_tipo = {'Crítica': 'red', 'Media': 'orange', 'Leve': 'green'}
    
    # Draw optimal routes
    for amb_id, ruta_info in rutas_optimas.items():
        asig = next(a for a in asignaciones if a['ambulancia_id'] == amb_id)
        tipo_emerg = asig['tipo_emergencia']
        color_ruta = colores_tipo[tipo_emerg]
        
        ruta_coords = [
            (G.nodes[nodo]['y'], G.nodes[nodo]['x'])
            for nodo in ruta_info['ruta']
        ]
        
        folium.PolyLine(
            locations=ruta_coords,
            color=color_ruta,
            weight=5,
            opacity=0.8,
            popup=f"""
            <b>{amb_id}</b><br>
            Tipo: {asig['ambulancia_tipo']}<br>
            Emergencia: {tipo_emerg}<br>
            Distancia: {ruta_info['distancia_km']:.2f} km<br>
            Tiempo: {asig['tiempo_min']:.1f} min<br>
            Velocidad req: {asig['velocidad_req']:.1f} km/h<br>
            Capacidad max: {G.edges[ruta_info['arcos'][0][:3]]['Capacidad_C']:.1f} km/h
            """
        ).add_to(m)
    
    # Mark origin
    folium.Marker(
        location=[lat_origen, lon_origen],
        popup=f"<b>🏥 BASE</b><br>Ambulancias: {len(asignaciones)}",
        icon=folium.Icon(color='blue', icon='home', prefix='fa')
    ).add_to(m)
    
    # Mark destinations
    iconos_tipo = {'Crítica': 'exclamation-triangle', 'Media': 'exclamation-circle', 'Leve': 'info-circle'}
    
    for asig in asignaciones:
        nodo_dest = asig['nodo_destino']
        tipo_emerg = asig['tipo_emergencia']
        
        lat_dest = G.nodes[nodo_dest]['y']
        lon_dest = G.nodes[nodo_dest]['x']
        
        folium.Marker(
            location=[lat_dest, lon_dest],
            popup=f"""
            <b>🚨 {tipo_emerg.upper()}</b><br>
            Atendido: {asig['ambulancia_id']}<br>
            Tiempo: {asig['tiempo_min']:.1f} min<br>
            Velocidad: {asig['velocidad_req']:.1f} km/h
            """,
            icon=folium.Icon(
                color=colores_tipo[tipo_emerg],
                icon=iconos_tipo[tipo_emerg],
                prefix='fa'
            )
        ).add_to(m)
    
    # Add legend
    leyenda_html = f"""
    <div style="
    position: fixed;
    bottom: 50px; left: 50px; width: 220px; height: 180px;
    background-color: white; border:2px solid grey; z-index:9999;
    font-size:14px; padding: 10px">
    <p style="margin:0; font-weight:bold;">🚑 Leyenda</p>
    <hr style="margin:5px 0;">
    <p style="margin:2px 0;"><span style="color:red;">━━</span> Crítica</p>
    <p style="margin:2px 0;"><span style="color:orange;">━━</span> Media</p>
    <p style="margin:2px 0;"><span style="color:green;">━━</span> Leve</p>
    <hr style="margin:5px 0;">
    <p style="margin:2px 0;">🏥 Base</p>
    <p style="margin:2px 0;">🚨 Incidentes</p>
    <hr style="margin:5px 0;">
    <p style="margin:2px 0; font-size:12px;">Costo: <b>${resultado['costo_total']:.2f}</b></p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(leyenda_html))
    
    return m

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'ambulancias' not in st.session_state:
    # Initialize with default fleet
    st.session_state.ambulancias = [
        Ambulancia("Amb_001", personal=3, equipamiento=5, insumos=10),   # Costo: 85 (Leve)
        Ambulancia("Amb_002", personal=5, equipamiento=8, insumos=15),   # Costo: 135 (Media)
        Ambulancia("Amb_003", personal=7, equipamiento=12, insumos=20),  # Costo: 190 (Media)
        Ambulancia("Amb_004", personal=10, equipamiento=15, insumos=25), # Costo: 250 (Crítica)
        Ambulancia("Amb_005", personal=2, equipamiento=3, insumos=5),    # Costo: 50 (Leve)
        Ambulancia("Amb_006", personal=4, equipamiento=6, insumos=12),   # Costo: 106 (Media)
        Ambulancia("Amb_007", personal=6, equipamiento=10, insumos=18),  # Costo: 164 (Media)
        Ambulancia("Amb_008", personal=8, equipamiento=14, insumos=22),  # Costo: 216 (Crítica)
        Ambulancia("Amb_009", personal=3, equipamiento=4, insumos=8),    # Costo: 74 (Leve)
        Ambulancia("Amb_010", personal=12, equipamiento=18, insumos=30), # Costo: 300 (Crítica)
        Ambulancia("Amb_011", personal=5, equipamiento=7, insumos=13),   # Costo: 124 (Media)
        Ambulancia("Amb_012", personal=2, equipamiento=5, insumos=7),    # Costo: 66 (Leve)
        Ambulancia("Amb_013", personal=7, equipamiento=9, insumos=16),   # Costo: 163 (Media)
        Ambulancia("Amb_014", personal=9, equipamiento=13, insumos=24),  # Costo: 227 (Crítica)
        Ambulancia("Amb_015", personal=1, equipamiento=2, insumos=4)     # Costo: 32 (Desconocido)
    ]

if 'G' not in st.session_state:
    st.session_state.G = None

if 'resultado' not in st.session_state:
    st.session_state.resultado = None

if 'R_k' not in st.session_state:
    st.session_state.R_k = None

# Default parameters
if 'num_incidentes' not in st.session_state:
    st.session_state.num_incidentes = 5

if 'R_MIN' not in st.session_state:
    st.session_state.R_MIN = 20

if 'R_MAX' not in st.session_state:
    st.session_state.R_MAX = 50

if 'C_MIN' not in st.session_state:
    st.session_state.C_MIN = 50

if 'C_MAX' not in st.session_state:
    st.session_state.C_MAX = 100

if 'FACTOR_RELAJACION' not in st.session_state:
    st.session_state.FACTOR_RELAJACION = 1.0

if 'ORIGEN' not in st.session_state:
    st.session_state.ORIGEN = None

if 'DESTINOS' not in st.session_state:
    st.session_state.DESTINOS = None

if 'incidentes_generados' not in st.session_state:
    st.session_state.incidentes_generados = False

# ============================================================================
# STREAMLIT UI
# ============================================================================

st.title("🚑 Ambulance Multi-Flow Optimization System")
st.markdown("Optimal ambulance dispatch using multi-commodity network flow optimization")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("📊 Network Parameters")
    
    st.session_state.R_MIN = st.number_input(
        "R_min (Min Required Speed, km/h)",
        min_value=10,
        max_value=50,
        value=st.session_state.R_MIN,
        step=5,
        help="Minimum required speed for ambulances"
    )
    
    st.session_state.R_MAX = st.number_input(
        "R_max (Max Required Speed, km/h)",
        min_value=st.session_state.R_MIN + 10,
        max_value=100,
        value=st.session_state.R_MAX,
        step=5,
        help="Maximum required speed for ambulances"
    )
    
    st.session_state.C_MIN = st.number_input(
        "C_min (Min Link Capacity, km/h)",
        min_value=20,
        max_value=100,
        value=st.session_state.C_MIN,
        step=5,
        help="Minimum capacity (max speed) of street links"
    )
    
    st.session_state.C_MAX = st.number_input(
        "C_max (Max Link Capacity, km/h)",
        min_value=st.session_state.C_MIN + 10,
        max_value=150,
        value=st.session_state.C_MAX,
        step=5,
        help="Maximum capacity (max speed) of street links"
    )
    
    st.session_state.FACTOR_RELAJACION = st.number_input(
        "Relaxation Factor",
        min_value=1.0,
        max_value=3.0,
        value=st.session_state.FACTOR_RELAJACION,
        step=0.1,
        help="Capacity relaxation multiplier (allows sum of speeds to exceed link capacity)"
    )
    
    st.divider()
    
    st.info(f"""
    **Fixed Parameters:**
    - Location: ({LATITUD_CENTRO}, {LONGITUD_CENTRO})
    - Radius: {RADIO_METROS}m
    - β (time weight): {BETA}
    - γ (cost weight): {GAMMA}
    
    **Current Relaxation:** {st.session_state.FACTOR_RELAJACION}x
    """)

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["🚑 Fleet Management", "🚨 Incidents", "🎯 Optimization", "📊 Results"])

# ============================================================================
# TAB 1: FLEET MANAGEMENT
# ============================================================================

with tab1:
    st.header("🚑 Ambulance Fleet Management")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Current Fleet")
        
        if st.session_state.ambulancias:
            df_ambulancias = pd.DataFrame([amb.to_dict() for amb in st.session_state.ambulancias])
            
            # Style the dataframe
            def color_tipo(val):
                colors = {'Crítica': 'background-color: #ffcccc',
                         'Media': 'background-color: #ffe6cc',
                         'Leve': 'background-color: #ccffcc'}
                return colors.get(val, '')
            
            styled_df = df_ambulancias.style.applymap(color_tipo, subset=['Tipo'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            # Fleet statistics
            tipos_count = Counter([amb.tipo for amb in st.session_state.ambulancias])
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Total Ambulances", len(st.session_state.ambulancias))
            col_b.metric("Critical", tipos_count.get('Crítica', 0))
            col_c.metric("Medium", tipos_count.get('Media', 0))
            col_d.metric("Light", tipos_count.get('Leve', 0))
        else:
            st.warning("No ambulances in fleet")
    
    with col2:
        st.subheader("Add New Ambulance")
        
        with st.form("add_ambulance_form"):
            new_id = st.text_input(
                "Ambulance ID",
                value=f"Amb_{len(st.session_state.ambulancias)+1:03d}"
            )
            
            new_personal = st.number_input(
                "Medical Staff",
                min_value=1,
                max_value=15,
                value=5,
                step=1
            )
            
            new_equipamiento = st.number_input(
                "Equipment",
                min_value=1,
                max_value=20,
                value=8,
                step=1
            )
            
            new_insumos = st.number_input(
                "Supplies",
                min_value=1,
                max_value=30,
                value=15,
                step=1
            )
            
            # Preview operational cost
            preview_cost = (new_personal * COEF_PERSONAL + 
                          new_equipamiento * COEF_EQUIPAMIENTO + 
                          new_insumos * COEF_INSUMOS)
            
            if preview_cost >= 200:
                preview_tipo = 'Crítica'
            elif preview_cost >= 100:
                preview_tipo = 'Media'
            elif preview_cost >= 50:
                preview_tipo = 'Leve'
            else:
                preview_tipo = 'Desconocido'
            
            st.info(f"**Preview:** Cost: ${preview_cost} → Type: {preview_tipo}")
            
            submitted = st.form_submit_button("➕ Add Ambulance", use_container_width=True)
            
            if submitted:
                # Check if ID already exists
                if any(amb.id == new_id for amb in st.session_state.ambulancias):
                    st.error(f"ID '{new_id}' already exists!")
                else:
                    nueva_amb = Ambulancia(new_id, new_personal, new_equipamiento, new_insumos)
                    st.session_state.ambulancias.append(nueva_amb)
                    st.success(f"✅ {new_id} added successfully!")
                    st.rerun()
    
    # Remove ambulance section
    if st.session_state.ambulancias:
        st.divider()
        st.subheader("Remove Ambulance")
        
        col_remove1, col_remove2 = st.columns([3, 1])
        
        with col_remove1:
            amb_to_remove = st.selectbox(
                "Select ambulance to remove:",
                options=[amb.id for amb in st.session_state.ambulancias],
                key="remove_selectbox"
            )
        
        with col_remove2:
            if st.button("🗑️ Remove", use_container_width=True, type="secondary"):
                st.session_state.ambulancias = [
                    amb for amb in st.session_state.ambulancias 
                    if amb.id != amb_to_remove
                ]
                st.success(f"Removed {amb_to_remove}")
                st.rerun()

# ============================================================================
# TAB 2: INCIDENTS
# ============================================================================

with tab2:
    st.header("🚨 Incident Management")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 Configuration")
        
        st.session_state.num_incidentes = st.number_input(
            "Number of Incidents",
            min_value=1,
            max_value=15,
            value=st.session_state.num_incidentes,
            step=1,
            help="Total number of emergency incidents to simulate"
        )
        
        st.divider()
        
        # Button to generate incidents
        if st.button("🔄 Generate Incidents", use_container_width=True, type="primary"):
            # Auto-load network if not loaded
            if st.session_state.G is None:
                with st.spinner("📡 Loading street network (first time)..."):
                    G = descargar_red(LATITUD_CENTRO, LONGITUD_CENTRO, RADIO_METROS)
                    if G is None:
                        st.error("❌ Failed to download network. Check your internet connection.")
                        st.stop()
                    st.session_state.G = G
                    st.success(f"✅ Network loaded: {len(G.nodes())} nodes")
            
            # Generate incidents
            with st.spinner("🔄 Generating incident locations and types..."):
                random.seed(int(time.time()))
                ORIGEN, DESTINOS = generar_origen_destinos(
                    st.session_state.G,
                    st.session_state.num_incidentes
                )
                st.session_state.ORIGEN = ORIGEN
                st.session_state.DESTINOS = DESTINOS
                st.session_state.incidentes_generados = True
                st.success(f"✅ Generated {len(DESTINOS)} incidents!")
                st.rerun()
        
        # Network status
        if st.session_state.G is None:
            st.info(f"""
            **Current Configuration:**
            - Requested Incidents: {st.session_state.num_incidentes}
            - Available Ambulances: {len(st.session_state.ambulancias)}
            - Location: Medellín, Colombia
            - Coverage Radius: {RADIO_METROS}m
            
            ℹ️ **Network will load automatically when you generate incidents**
            """)
        else:
            st.success(f"""
            **Current Configuration:**
            - Requested Incidents: {st.session_state.num_incidentes}
            - Available Ambulances: {len(st.session_state.ambulancias)}
            - Location: Medellín, Colombia
            - Coverage Radius: {RADIO_METROS}m
            - Network: {len(st.session_state.G.nodes())} nodes, {len(st.session_state.G.edges())} edges ✅
            """)
    
    with col2:
        st.subheader("📊 Current Incidents")
        
        if not st.session_state.incidentes_generados or st.session_state.DESTINOS is None:
            st.warning("⚠️ No incidents generated yet. Click **Generate Incidents** button.")
        else:
            # Count incidents by type
            tipos_incidentes = Counter(st.session_state.DESTINOS.values())
            
            # Display metrics
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("🔴 Critical", tipos_incidentes.get('Crítica', 0))
            col_b.metric("🟠 Medium", tipos_incidentes.get('Media', 0))
            col_c.metric("🟢 Light", tipos_incidentes.get('Leve', 0))
            
            st.metric("**Total Generated**", len(st.session_state.DESTINOS))
            
            st.divider()
            
            # Show incidents table
            st.subheader("📍 Incident Details")
            
            incidentes_data = []
            for nodo, tipo in st.session_state.DESTINOS.items():
                incidentes_data.append({
                    'Node ID': nodo,
                    'Type': tipo,
                    'Priority': {'Crítica': 3, 'Media': 2, 'Leve': 1}[tipo]
                })
            
            df_incidentes = pd.DataFrame(incidentes_data)
            df_incidentes = df_incidentes.sort_values('Priority', ascending=False)
            
            # Style the dataframe
            def color_tipo_incident(row):
                colors = {
                    'Crítica': 'background-color: #ffcccc',
                    'Media': 'background-color: #ffe6cc',
                    'Leve': 'background-color: #ccffcc'
                }
                return [colors.get(row['Type'], '')] * len(row)
            
            styled_df_inc = df_incidentes.style.apply(color_tipo_incident, axis=1)
            st.dataframe(styled_df_inc, use_container_width=True, hide_index=True)
    
    st.divider()
    
    st.subheader("ℹ️ Emergency Type Information")
    
    col_info1, col_info2, col_info3 = st.columns(3)
    
    with col_info1:
        st.markdown("""
        **🔴 Critical**
        - Highest priority
        - Fastest response required
        - Requires advanced equipment
        - Highest speed requirement
        """)
    
    with col_info2:
        st.markdown("""
        **🟠 Medium**
        - Standard emergency
        - Moderate response time
        - Standard equipment
        - Medium speed requirement
        """)
    
    with col_info3:
        st.markdown("""
        **🟢 Light**
        - Lower priority
        - Flexible response time
        - Basic equipment
        - Lower speed requirement
        """)

# ============================================================================
# TAB 3: OPTIMIZATION
# ============================================================================

with tab3:
    st.header("🎯 Optimization Control")
    
    # Load network if not loaded
    if st.session_state.G is None:
        with st.spinner("📡 Downloading street network..."):
            G = descargar_red(LATITUD_CENTRO, LONGITUD_CENTRO, RADIO_METROS)
            if G is not None:
                st.session_state.G = G
                st.success(f"✅ Network loaded: {len(G.nodes())} nodes, {len(G.edges())} edges")
            else:
                st.error("❌ Failed to load network")
    else:
        st.info(f"✅ Network ready: {len(st.session_state.G.nodes())} nodes, {len(st.session_state.G.edges())} edges")
    
    # Show incident status
    if st.session_state.incidentes_generados and st.session_state.DESTINOS is not None:
        tipos_count = Counter(st.session_state.DESTINOS.values())
        st.success(f"""
        ✅ **Incidents Ready:** {len(st.session_state.DESTINOS)} incidents generated
        - 🔴 Critical: {tipos_count.get('Crítica', 0)} | 🟠 Medium: {tipos_count.get('Media', 0)} | 🟢 Light: {tipos_count.get('Leve', 0)}
        """)
    else:
        st.warning("⚠️ No incidents generated. Go to **Incidents** tab to generate them first.")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Recalculate Capacities", use_container_width=True, type="secondary"):
            if st.session_state.G is not None:
                with st.spinner("🔄 Recalculating link capacities..."):
                    random.seed(int(time.time()))
                    R_k = asignar_capacidades_velocidades(
                        st.session_state.G,
                        st.session_state.C_MIN,
                        st.session_state.C_MAX,
                        st.session_state.R_MIN,
                        st.session_state.R_MAX
                    )
                    st.session_state.R_k = R_k
                    st.success("✅ Capacities recalculated!")
                    
                    # Show speed requirements
                    st.subheader("Speed Requirements by Type")
                    for tipo, vel in sorted(R_k.items(), key=lambda x: x[1]):
                        st.metric(f"{tipo}", f"{vel:.1f} km/h")
    
    with col2:
        if st.button("⚡ Recalculate Flows (Run Optimization)", use_container_width=True, type="primary"):
            if st.session_state.G is None:
                st.error("❌ Network not loaded!")
            elif len(st.session_state.ambulancias) == 0:
                st.error("❌ No ambulances in fleet!")
            elif not st.session_state.incidentes_generados or st.session_state.DESTINOS is None:
                st.error("❌ No incidents generated! Go to Incidents tab and generate incidents first.")
            else:
                with st.spinner("🔄 Running optimization model..."):
                    # Use generated incidents from session state
                    ORIGEN = st.session_state.ORIGEN
                    DESTINOS = st.session_state.DESTINOS
                    
                    # Assign capacities if not done
                    if st.session_state.R_k is None:
                        R_k = asignar_capacidades_velocidades(
                            st.session_state.G,
                            st.session_state.C_MIN,
                            st.session_state.C_MAX,
                            st.session_state.R_MIN,
                            st.session_state.R_MAX
                        )
                        st.session_state.R_k = R_k
                    
                    # Solve optimization
                    resultado, error = resolver_optimizacion(
                        st.session_state.G,
                        st.session_state.ambulancias,
                        ORIGEN,
                        DESTINOS,
                        st.session_state.R_k,
                        st.session_state.FACTOR_RELAJACION
                    )
                    
                    if error:
                        st.error(f"❌ Optimization failed: {error}")
                    else:
                        st.session_state.resultado = resultado
                        st.success("✅ Optimization completed successfully!")
                        st.balloons()

# ============================================================================
# TAB 4: RESULTS
# ============================================================================

with tab4:
    st.header("📊 Optimization Results")
    
    if st.session_state.resultado is None:
        st.info("👈 Run optimization in the **Optimization** tab to see results")
    else:
        resultado = st.session_state.resultado
        
        # Summary metrics
        st.subheader("📈 Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Total Cost", f"${resultado['costo_total']:.2f}")
        col2.metric("Dispatched", len(resultado['asignaciones']))
        col3.metric("Available", len(st.session_state.ambulancias) - len(resultado['asignaciones']))
        
        avg_time = sum([a['tiempo_min'] for a in resultado['asignaciones']]) / len(resultado['asignaciones'])
        col4.metric("Avg Response Time", f"{avg_time:.1f} min")
        
        st.divider()
        
        # Assignments table
        st.subheader("🚑 Ambulance Assignments")
        
        df_asignaciones = pd.DataFrame(resultado['asignaciones'])
        df_asignaciones = df_asignaciones.sort_values('prioridad', ascending=False)
        
        # Format columns
        df_display = df_asignaciones[[
            'ambulancia_id', 'ambulancia_tipo', 'tipo_emergencia', 
            'tiempo_min', 'distancia_km', 'velocidad_req', 'costo_operativo'
        ]].copy()
        
        df_display.columns = [
            'Ambulance', 'Amb Type', 'Emergency', 
            'Time (min)', 'Distance (km)', 'Speed (km/h)', 'Op Cost'
        ]
        
        df_display['Time (min)'] = df_display['Time (min)'].round(2)
        df_display['Distance (km)'] = df_display['Distance (km)'].round(3)
        df_display['Speed (km/h)'] = df_display['Speed (km/h)'].round(1)
        
        st.dataframe(df_display, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # Interactive map
        st.subheader("🗺️ Route Visualization")
        
        with st.spinner("🗺️ Generating map..."):
            mapa = crear_mapa(st.session_state.G, resultado)
            
            # Save map to HTML string
            from io import BytesIO
            import base64
            
            # Convert map to HTML
            map_html = mapa._repr_html_()
            
            # Display using components.html
            import streamlit.components.v1 as components
            components.html(map_html, height=600, scrolling=True)
        
        st.divider()
        
        # Detailed route information
        st.subheader("🛣️ Detailed Routes")
        
        for asig in resultado['asignaciones']:
            amb_id = asig['ambulancia_id']
            ruta_info = resultado['rutas'][amb_id]
            
            with st.expander(f"📍 {amb_id} → {asig['tipo_emergencia']} Emergency"):
                col_a, col_b, col_c = st.columns(3)
                
                col_a.metric("Distance", f"{ruta_info['distancia_km']:.3f} km")
                col_b.metric("Time", f"{asig['tiempo_min']:.2f} min")
                col_c.metric("Required Speed", f"{asig['velocidad_req']:.1f} km/h")
                
                st.write("**Route Path:**")
                st.code(" → ".join(map(str, ruta_info['ruta'][:10])) + 
                       (" → ..." if len(ruta_info['ruta']) > 10 else ""))
                
                st.write(f"**Segments:** {len(ruta_info['arcos'])}")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>🚑 Ambulance Multi-Flow Optimization System</p>
    <p>Powered by PuLP, OSMnx, and Streamlit</p>
</div>
""", unsafe_allow_html=True)
