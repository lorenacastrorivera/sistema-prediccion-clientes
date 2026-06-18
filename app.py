
import streamlit as st
import pandas as pd

# ==================================
# CONFIGURACIÓN
# ==================================
st.set_page_config(
    page_title="Sistema Inteligente de Predicción",
    layout="wide"
)
st.error("🚨 VERSION NUEVA APP.PY")
# ==================================
# MENÚ
# ==================================
menu = st.radio(
    "Seleccione una opción",
    [
        "🏠 Inicio",
        "📊 Resultados",
        "🔮 Simulación de Escenarios"
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
        "Herramienta desarrollada para visualizar y consumir modelos predictivos."
    )

# ==================================
# RESULTADOS
# ==================================
if menu == "📊 Resultados":

    st.title("📊 Resultados de los Modelos")

    df = pd.read_excel(
        "resultados_timeseriessplit.xlsx"
    )

    st.dataframe(
        df,
        use_container_width=True
    )

    st.subheader("Ranking por R²")

    df_graf = df.sort_values(
        by="R2_Promedio",
        ascending=False
    )

    st.bar_chart(
        data=df_graf,
        x="Modelo",
        y="R2_Promedio"
    )

# ==================================
# SIMULACIÓN
# ==================================
if menu == "🔮 Simulación de Escenarios":

    st.title("🔮 Simulación de Escenarios")

    st.info(
        "Próximamente se incorporará la predicción automática utilizando los siete modelos entrenados."
    )

    st.subheader("Variables comerciales")

    decos = st.number_input(
        "Decodificadores promedio",
        value=2.0
    )

    mensualidad = st.number_input(
        "Mensualidad promedio",
        value=140.0
    )

    programacion = st.selectbox(
        "Programación",
        [
            "BRONCE HD",
            "PLATA HD",
            "ORO HD"
        ]
    )

    mod_pago = st.selectbox(
        "Modalidad de pago",
        [
            "EFECTIVO",
            "TC"
        ]
    )

    estado = st.selectbox(
        "Estado de cuenta",
        [
            "NORMAL",
            "FIRST REMINDER",
            "COLLECTION"
        ]
    )

    st.subheader("Historial reciente")

    lag1 = st.number_input(
        "Clientes semana anterior",
        value=20
    )

    lag3 = st.number_input(
        "Clientes hace 3 semanas",
        value=18
    )

    lag6 = st.number_input(
        "Clientes hace 6 semanas",
        value=15
    )

    rolling3 = (lag1 + lag3 + lag6) / 3

    st.metric(
        "Rolling 3 calculado",
        round(rolling3, 2)
    )
