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

    ### Modelo de referencia

    **Regresión Lineal Bayesiana**
    """)

    st.divider()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Observaciones",
        "186"
    )

    col2.metric(
        "Modelos Evaluados",
        "7"
    )

    col3.metric(
        "Modelo de referencia",
        "Regresión Lineal Bayesiana"
    )

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
        "resultados_kfold_definitivos_186_observaciones.xlsx"
    )

    # Limpiar espacios en nombres de columnas
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    # Mostrar tabla
    st.dataframe(
        df,
        use_container_width=True
    )

    # --------------------------------------------------------
    # Verificar columnas requeridas
    # --------------------------------------------------------

    columnas_requeridas = [
        "MODELO",
        "R2_PROMEDIO",
        "MSE_PROMEDIO",
        "MAE_PROMEDIO"
    ]

    columnas_faltantes = [
        columna
        for columna in columnas_requeridas
        if columna not in df.columns
    ]

    if columnas_faltantes:

        st.error(
            "El archivo de resultados no contiene las columnas "
            "esperadas."
        )

        st.write(
            "Columnas encontradas en el archivo:"
        )

        st.write(
            df.columns.tolist()
        )

        st.stop()

    # --------------------------------------------------------
    # IDENTIFICAR RESULTADOS DESTACADOS
    # --------------------------------------------------------

    mejor_r2 = df.loc[
        df["R2_PROMEDIO"].idxmax()
    ]

    mejor_mse = df.loc[
        df["MSE_PROMEDIO"].idxmin()
    ]

    mejor_mae = df.loc[
        df["MAE_PROMEDIO"].idxmin()
    ]

    # --------------------------------------------------------
    # RESUMEN
    # --------------------------------------------------------

    st.subheader("📊 Resumen del desempeño")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Mayor R²",
            mejor_r2["MODELO"]
        )

    with col2:
        st.metric(
            "Menor MSE",
            mejor_mse["MODELO"]
        )

    with col3:
        st.metric(
            "Menor MAE",
            mejor_mae["MODELO"]
        )

    st.info(
        "La Regresión Cuantil presenta el mayor R² promedio "
        "y el menor MAE promedio, mientras que la Regresión "
        "Lineal Bayesiana presenta el menor MSE promedio."
    )

    st.divider()

    # --------------------------------------------------------
    # COMPARACIÓN SEGÚN R²
    # --------------------------------------------------------

    st.subheader(
        "📈 Comparación de Modelos según R²"
    )

    df_graf = df.sort_values(
        by="R2_PROMEDIO",
        ascending=False
    )

    st.bar_chart(
        data=df_graf,
        x="MODELO",
        y="R2_PROMEDIO"
    )

    # --------------------------------------------------------
    # COMPARACIÓN SEGÚN MSE
    # --------------------------------------------------------

    st.subheader(
        "📉 Comparación de Modelos según MSE"
    )

    df_mse = df.sort_values(
        by="MSE_PROMEDIO",
        ascending=True
    )

    st.bar_chart(
        data=df_mse,
        x="MODELO",
        y="MSE_PROMEDIO"
    )

    # --------------------------------------------------------
    # COMPARACIÓN SEGÚN MAE
    # --------------------------------------------------------

    st.subheader(
        "📉 Comparación de Modelos según MAE"
    )

    df_mae = df.sort_values(
        by="MAE_PROMEDIO",
        ascending=True
    )

    st.bar_chart(
        data=df_mae,
        x="MODELO",
        y="MAE_PROMEDIO"
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
            "Clientes en la observación anterior (LAG 1)",
            min_value=0,
            value=20,
            step=1
        )

        lag3 = st.number_input(
            "Clientes en la tercera observación anterior (LAG 3)",
            min_value=0,
            value=18,
            step=1
        )

        lag6 = st.number_input(
            "Clientes en la sexta observación anterior (LAG 6)",
            min_value=0,
            value=15,
            step=1
        )

    with col2:

        st.markdown("### 📊 Promedio móvil")

        # Rolling 3:
        # utiliza las tres observaciones anteriores

        lag_rolling_2 = st.number_input(
            "Clientes en la segunda observación anterior",
            min_value=0,
            value=19,
            step=1
        )

        rolling3 = (
            lag1 +
            lag_rolling_2 +
            lag3
        ) / 3

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
            # Cargar modelo de referencia
            # ------------------------------------------------

            modelo = joblib.load(
                "modelo_regresion_lineal_bayesiana_multiple.pkl"
            )


            # ------------------------------------------------
            # Crear entrada
            # ------------------------------------------------

            entrada = pd.DataFrame({

                "DECOS_PROM_LAG1": [
                    decos
                ],

                "MENSUALIDAD_PROM_LAG1": [
                    mensualidad
                ],

                "PROGRAMACION_MAS_FRECUENTE_LAG1": [
                    programacion
                ],

                "MOD_PAGO_MAS_FRECUENTE_LAG1": [
                    mod_pago
                ],

                "ESTADO_CUENTA_MAS_FRECUENTE_LAG1": [
                    estado
                ],

                "CLIENTES_LAG1": [
                    lag1
                ],

                "CLIENTES_LAG3": [
                    lag3
                ],

                "CLIENTES_LAG6": [
                    lag6
                ],

                "CLIENTES_ROLLING3": [
                    rolling3
                ]

            })


            # ------------------------------------------------
            # Generar predicción
            # ------------------------------------------------

            prediccion = modelo.predict(
                entrada
            )[0]


            # ------------------------------------------------
            # Adecuar resultado a cantidad de clientes
            # ------------------------------------------------

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
                "**Modelo utilizado:** Regresión Lineal Bayesiana"
            )

        except Exception as e:

            st.error(
                "Ocurrió un error al generar la predicción."
            )

            st.exception(e)
