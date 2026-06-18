import streamlit as st
import streamlit as st
import pandas as pd
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
