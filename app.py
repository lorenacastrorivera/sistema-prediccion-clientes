import streamlit as st
import pandas as pd
import joblib

# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Sistema Inteligente de Predicción",
    layout="wide"
)

# ============================================================
# MENÚ
# ============================================================

menu = st.radio(
    "Seleccione una opción",
    [
        "🏠 Inicio",
        "📊 Resultados",
        "🔮 Simulación"
    ]
)

# ============================================================
# INICIO
# ============================================================

if menu == "🏠 Inicio":

    st.title(
        "📈 Sistema Inteligente de Predicción del Crecimiento de Clientes"
    )

    st.markdown("""
    ### Investigación

    Técnicas de aprendizaje supervisado para predecir el crecimiento
    de clientes de televisión por paga.

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

    col1.metric("Observaciones", "184")
    col2.metric("Modelos Evaluados", "7")
    col3.metric("Mejor Modelo", "Regresión Lineal Bayesiana")

    st.success(
        "Sistema desarrollado para consumir modelos predictivos "
        "de crecimiento de clientes."
    )


# ============================================================
# RESULTADOS
# ============================================================

elif menu == "📊 Resultados":

    st.title("📊 Resultados de los Modelos")

    # --------------------------------------------------------
    # Cargar resultados definitivos
    # --------------------------------------------------------

    df = pd.read_excel(
        "resultados_kfold_sin_leakage_definitivo.xlsx"
    )

    st.dataframe(
        df,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # Ranking según R²
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Ranking según MSE
    # --------------------------------------------------------

    st.subheader("Ranking de Modelos según MSE")

    df_mse = df.sort_values(
        by="MSE_Promedio",
        ascending=True
    )

    st.bar_chart(
        data=df_mse,
        x="Modelo",
        y="MSE_Promedio"
    )

    # --------------------------------------------------------
    # Ranking según MAE
    # --------------------------------------------------------

    st.subheader("Ranking de Modelos según MAE")

    df_mae = df.sort_values(
        by="MAE_Promedio",
        ascending=True
    )

    st.bar_chart(
        data=df_mae,
        x="Modelo",
        y="MAE_Promedio"
    )


# ============================================================
# SIMULACIÓN
# ============================================================

elif menu == "🔮 Simulación":

    st.title("🔮 Simulación de Escenarios")

    st.info(
        "Ingrese valores comerciales e históricos para estimar "
        "la cantidad de nuevos clientes del siguiente periodo."
    )

    # ========================================================
    # VARIABLES COMERCIALES
    # ========================================================

    st.subheader("📋 Variables comerciales")

    col1, col2 = st.columns(2)

    with col1:

        decos = st.number_input(
            "Decodificadores promedio",
            min_value=0.0,
            value=2.0,
            step=0.1
        )

    with col2:

        mensualidad = st.number_input(
            "Mensualidad promedio",
            min_value=0.0,
            value=140.0,
            step=1.0
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

    # ========================================================
    # HISTORIAL
    # ========================================================

    st.subheader("📈 Historial reciente")

    col1, col2 = st.columns(2)

    with col1:

        lag1 = st.number_input(
            "Clientes en la observación anterior",
            min_value=0,
            value=20,
            step=1
        )

        lag2 = st.number_input(
            "Clientes en la segunda observación anterior",
            min_value=0,
            value=19,
            step=1
        )

        lag3 = st.number_input(
            "Clientes en la tercera observación anterior",
            min_value=0,
            value=18,
            step=1
        )

    with col2:

        lag6 = st.number_input(
            "Clientes en la sexta observación anterior",
            min_value=0,
            value=15,
            step=1
        )

        # Rolling 3 correcto
        rolling3 = (lag1 + lag2 + lag3) / 3

        st.metric(
            "Promedio móvil (Rolling 3)",
            round(rolling3, 2)
        )

    st.divider()

    # ========================================================
    # PREDICCIÓN
    # ========================================================

    if st.button(
        "🚀 Generar Predicción",
        use_container_width=True
    ):

        try:

            # ------------------------------------------------
            # Cargar modelo definitivo
            # ------------------------------------------------

            modelo = joblib.load(
                "modelo_regresion_lineal_bayesiana_multiple.pkl"
            )

            # ------------------------------------------------
            # Crear entrada
            # ------------------------------------------------

            entrada = pd.DataFrame({

                "DECOS_PROM_LAG1": [decos],

                "MENSUALIDAD_PROM_LAG1": [mensualidad],

                "PROGRAMACION_MAS_FRECUENTE_LAG1": [
                    programacion
                ],

                "MOD_PAGO_MAS_FRECUENTE_LAG1": [
                    mod_pago
                ],

                "ESTADO_CUENTA_MAS_FRECUENTE_LAG1": [
                    estado
                ],

                "CLIENTES_LAG1": [lag1],

                "CLIENTES_LAG3": [lag3],

                "CLIENTES_LAG6": [lag6],

                "CLIENTES_ROLLING3": [rolling3]
            })

            # ------------------------------------------------
            # Generar predicción
            # ------------------------------------------------

            prediccion = modelo.predict(entrada)[0]

            # Evitar resultados negativos
            prediccion_final = max(
                0,
                int(round(prediccion))
            )

            # ------------------------------------------------
            # Mostrar resultado
            # ------------------------------------------------

            st.success(
                "Predicción generada correctamente."
            )

            st.metric(
                "👥 Clientes Predichos",
                prediccion_final
            )

            st.info(
                "El resultado representa la cantidad estimada "
                "de nuevos clientes para el siguiente periodo."
            )

            st.write(
                "**Modelo utilizado:** "
                "Regresión Lineal Bayesiana"
            )

        except Exception as e:

            st.error(
                "Ocurrió un error al generar la predicción."
            )

            st.exception(e)
