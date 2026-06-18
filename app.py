import streamlit as st

# ==================================
# CONFIGURACIÓN
# ==================================
st.set_page_config(
    page_title="Sistema Inteligente de Predicción",
    layout="wide"
)

# ==================================
# MENÚ
# ==================================
menu = st.radio(
    "Seleccione una opción",
    [
        "🏠 Inicio",
        "📊 Resultados",
        "🔮 Simulación"
    ]
)

# ==================================
# INICIO
# ==================================
if menu == "🏠 Inicio":

    st.title(
        "📈 Sistema Inteligente de Predicción del Crecimiento de Clientes"
    )

    st.markdown("""
    ### Investigación

    Técnicas de aprendizaje supervisado para predecir el crecimiento de clientes.

    ### Aplicabilidad

    Esta metodología puede adaptarse a:

    - Empresas de telecomunicaciones
    - Empresas de agua potable
    - Empresas de energía eléctrica
    - Empresas de internet
    - Otros servicios
    """)

    st.divider()

    col1, col2, col3 = st.columns(3)

    col1.metric("Observaciones", "184")
    col2.metric("Modelos", "7")
    col3.metric("Mejor Modelo", "Bayesiana")

    st.success(
        "Aplicación funcionando correctamente."
    )

# ==================================
# RESULTADOS
# ==================================
if menu == "📊 Resultados":

    st.title("📊 Resultados")
    st.info("Página en construcción")

# ==================================
# SIMULACIÓN
# ==================================
if menu == "🔮 Simulación":

    st.title("🔮 Simulación")
    st.info("Página en construcción")
