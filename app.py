import streamlit as st
import pandas as pd
import joblib

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

    Técnicas de aprendizaje supervisado para predecir el crecimiento de clientes de televisión por paga.

    ### Aplicabilidad

    Esta metodología puede adaptarse a:

    - Empresas de telecomunicaciones
    - Empresas de agua potable
    - Empresas de energía eléctrica
    - Empresas de internet
    - Otros servicios

    ### Modelo seleccionado

    Regresión Lineal Bayesiana
    """)

    st.divider()

    col1, col2, col3 = st.columns(3)

    col1.metric("Observaciones", "42")
    col2.metric("Modelos Evaluados", "7")
    col3.metric("Mejor Modelo", "Bayesiana")

    st.success(
        "Sistema desarrollado para consumir modelos predictivos de crecimiento de clientes."
    )

# ==================================
# RESULTADOS
# ==================================
elif menu == "📊 Resultados":

    st.title("📊 Resultados de los Modelos")

    df = pd.read_excel(
        "resultados_timeseriessplit.xlsx"
    )

    st.dataframe(
        df,
        use_container_width=True
    )

    st.subheader("Ranking de Modelos según R²")

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
elif menu == "🔮 Simulación":

    st.title("🔮 Simulación de Escenarios")

    st.info(
        "Ingrese valores comerciales e históricos para estimar el crecimiento de clientes."
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
        "Clientes periodo anterior",
        value=20
    )

    lag3 = st.number_input(
        "Clientes hace 3 periodos",
        value=18
    )

    lag6 = st.number_input(
        "Clientes hace 6 periodos",
        value=15
    )

    rolling3 = (lag1 + lag3 + lag6) / 3

    st.metric(
        "Promedio móvil (Rolling 3)",
        round(rolling3, 2)
    )

    st.divider()

    if st.button("🚀 Generar Predicción"):

        modelo = joblib.load(
            "modelo_regresion_lineal_bayesiana_multiple.pkl"
        )

        entrada = pd.DataFrame({
            "DECOS_PROM": [decos],
            "MENSUALIDAD_PROM": [mensualidad],
            "PROGRAMACION_MAS_FRECUENTE": [programacion],
            "MOD_PAGO_MAS_FRECUENTE": [mod_pago],
            "ESTADO_CUENTA_MAS_FRECUENTE": [estado],
            "CLIENTES_LAG1": [lag1],
            "CLIENTES_LAG3": [lag3],
            "CLIENTES_LAG6": [lag6],
            "CLIENTES_ROLLING3": [rolling3]
        })

        prediccion = modelo.predict(entrada)[0]

        st.success(
            "Predicción generada correctamente utilizando el modelo de Regresión Lineal Bayesiana."
        )

        st.metric(
            "Clientes Predichos",
            int(round(prediccion))
        )

        st.info(
            "El resultado representa la cantidad estimada de nuevos clientes para el siguiente periodo."
        )

        st.write(
            "**Modelo utilizado:** Regresión Lineal Bayesiana"
        )
    
