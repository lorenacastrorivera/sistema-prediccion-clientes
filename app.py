import streamlit as st
import pandas as pd
import joblib
import os
import glob


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Sistema Inteligente de Predicción",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# CONFIGURACIÓN GENERAL DEL MODELO
# ============================================================

MODELO_PRINCIPAL = "modelo_regresion_lineal_bayesiana_multiple.pkl"

NOMBRE_MODELO = "Regresión Lineal Bayesiana"

VARIABLES_MODELO = [
    "DECOS_PROM_LAG1",
    "MENSUALIDAD_PROM_LAG1",
    "PROGRAMACION_MAS_FRECUENTE_LAG1",
    "MOD_PAGO_MAS_FRECUENTE_LAG1",
    "ESTADO_CUENTA_MAS_FRECUENTE_LAG1",
    "CLIENTES_LAG1",
    "CLIENTES_LAG3",
    "CLIENTES_LAG6",
    "CLIENTES_ROLLING3"
]


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def buscar_archivo(patrones):
    """
    Busca un archivo en el directorio de la aplicación.
    Permite trabajar aunque el nombre del Excel tenga
    pequeñas diferencias.
    """

    for patron in patrones:
        archivos = glob.glob(patron)

        if archivos:
            return archivos[0]

    return None


def cargar_modelo():
    """
    Carga el modelo de Regresión Lineal Bayesiana.
    """

    ruta = MODELO_PRINCIPAL

    if not os.path.exists(ruta):
        return None

    try:
        return joblib.load(ruta)

    except Exception as e:
        st.error(
            f"No fue posible cargar el modelo "
            f"{MODELO_PRINCIPAL}."
        )
        st.exception(e)
        return None


def obtener_nombre_modelo(modelo):
    """
    Obtiene el nombre del modelo cargado.
    """

    if modelo is None:
        return NOMBRE_MODELO

    nombre = type(modelo).__name__

    return nombre


def preparar_entrada(
    decos,
    mensualidad,
    programacion,
    mod_pago,
    estado,
    lag1,
    lag3,
    lag6,
    rolling3
):
    """
    Construye el DataFrame de entrada.
    """

    entrada = pd.DataFrame({
        "DECOS_PROM_LAG1": [decos],
        "MENSUALIDAD_PROM_LAG1": [mensualidad],
        "PROGRAMACION_MAS_FRECUENTE_LAG1": [programacion],
        "MOD_PAGO_MAS_FRECUENTE_LAG1": [mod_pago],
        "ESTADO_CUENTA_MAS_FRECUENTE_LAG1": [estado],
        "CLIENTES_LAG1": [lag1],
        "CLIENTES_LAG3": [lag3],
        "CLIENTES_LAG6": [lag6],
        "CLIENTES_ROLLING3": [rolling3]
    })

    return entrada


def preparar_entrada_para_modelo(modelo, entrada):
    """
    Intenta adaptar la entrada a la estructura esperada
    por el modelo.

    Si el modelo fue guardado como Pipeline, se envía
    directamente el DataFrame.

    Si el modelo posee feature_names_in_, se intenta
    respetar el orden de variables.
    """

    # --------------------------------------------------------
    # Si es un Pipeline, normalmente contiene internamente
    # el procesamiento de variables.
    # --------------------------------------------------------

    if hasattr(modelo, "steps"):
        return entrada

    # --------------------------------------------------------
    # Modelos que guardan los nombres de variables utilizadas
    # durante el entrenamiento.
    # --------------------------------------------------------

    if hasattr(modelo, "feature_names_in_"):

        nombres = list(modelo.feature_names_in_)

        # Si las variables coinciden exactamente
        if all(variable in entrada.columns for variable in nombres):

            return entrada[nombres]

    return entrada


def generar_prediccion(modelo, entrada):
    """
    Genera la predicción.

    Primero intenta utilizar directamente el DataFrame.
    """

    entrada_modelo = preparar_entrada_para_modelo(
        modelo,
        entrada
    )

    prediccion = modelo.predict(entrada_modelo)

    return prediccion[0]


def cargar_resultados_kfold():
    """
    Busca y carga el archivo actualizado de resultados K-Fold.
    """

    archivo = buscar_archivo([
        "resultados_kfold_folds_definitivos_*.xlsx",
        "resultados_kfold_folds_definitivos*.xlsx",
        "resultados_kfold_sin_leakage_definitivo.xlsx",
        "resultados_kfold*.xlsx"
    ])

    if archivo is None:
        return None, None

    try:
        return pd.read_excel(archivo), archivo

    except Exception:
        return None, archivo


def cargar_resultados_timeseries():
    """
    Busca y carga el archivo actualizado de resultados
    TimeSeriesSplit.
    """

    archivo = buscar_archivo([
        "resultados_timeseriesplit_definitivos_*.xlsx",
        "resultados_timeseriesplit_definitivos*.xlsx",
        "resultados_timeseriesplit*.xlsx"
    ])

    if archivo is None:
        return None, None

    try:
        return pd.read_excel(archivo), archivo

    except Exception:
        return None, archivo


def identificar_columnas(df):
    """
    Identifica las columnas principales de métricas.
    """

    if df is None:
        return None, None, None

    columnas = df.columns.tolist()

    r2 = None
    mse = None
    mae = None

    for columna in columnas:

        nombre = str(columna).upper()

        if "R2" in nombre and "PROM" in nombre:
            r2 = columna

        elif "MSE" in nombre and "PROM" in nombre:
            mse = columna

        elif "MAE" in nombre and "PROM" in nombre:
            mae = columna

    return r2, mse, mae


def mostrar_resumen_resultados(df):
    """
    Muestra resumen de las métricas disponibles.
    """

    if df is None or df.empty:
        return

    r2_col, mse_col, mae_col = identificar_columnas(df)

    if "Modelo" not in df.columns:
        return

    st.subheader("📊 Resumen del desempeño")

    columnas = []

    if r2_col is not None:
        columnas.append("r2")

    if mse_col is not None:
        columnas.append("mse")

    if mae_col is not None:
        columnas.append("mae")

    if not columnas:
        return

    n_columnas = len(columnas)

    cols = st.columns(n_columnas)

    indice = 0

    # --------------------------------------------------------
    # R²
    # --------------------------------------------------------

    if r2_col is not None:

        fila = df.loc[df[r2_col].idxmax()]

        with cols[indice]:

            st.metric(
                "Mayor R²",
                str(fila["Modelo"])
            )

            st.caption(
                f"{r2_col}: "
                f"{fila[r2_col]:.4f}"
            )

        indice += 1

    # --------------------------------------------------------
    # MSE
    # --------------------------------------------------------

    if mse_col is not None:

        fila = df.loc[df[mse_col].idxmin()]

        with cols[indice]:

            st.metric(
                "Menor MSE",
                str(fila["Modelo"])
            )

            st.caption(
                f"{mse_col}: "
                f"{fila[mse_col]:.4f}"
            )

        indice += 1

    # --------------------------------------------------------
    # MAE
    # --------------------------------------------------------

    if mae_col is not None:

        fila = df.loc[df[mae_col].idxmin()]

        with cols[indice]:

            st.metric(
                "Menor MAE",
                str(fila["Modelo"])
            )

            st.caption(
                f"{mae_col}: "
                f"{fila[mae_col]:.4f}"
            )


def normalizar_nombre_modelo(nombre):
    """
    Normaliza algunos nombres de modelos para mostrarlos
    de manera más amigable.
    """

    nombre = str(nombre)

    equivalencias = {

        "LinearRegression":
            "Regresión Lineal Múltiple",

        "BayesianRidge":
            "Regresión Lineal Bayesiana",

        "PoissonRegressor":
            "Regresión Poisson",

        "QuantileRegressor":
            "Regresión Cuantil",

        "RandomForestRegressor":
            "Random Forest",

        "GradientBoostingRegressor":
            "Gradient Boosting",

        "MLPRegressor":
            "Red Neuronal MLP"
    }

    return equivalencias.get(
        nombre,
        nombre
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
    ],
    horizontal=True
)


# ============================================================
# INICIO
# ============================================================

if menu == "🏠 Inicio":

    st.title(
        "📈 Sistema Inteligente de Predicción "
        "del Crecimiento de Clientes"
    )

    st.markdown(
        """
        ### Investigación

        Aplicación de técnicas de aprendizaje supervisado
        para predecir el crecimiento de clientes de
        televisión por paga.

        ### Aplicabilidad

        Esta metodología puede adaptarse a:

        - Empresas de telecomunicaciones
        - Empresas de agua potable
        - Empresas de energía eléctrica
        - Empresas de internet
        - Otros servicios

        ### Modelo utilizado para la simulación

        **Regresión Lineal Bayesiana**
        """
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Observaciones",
            "186"
        )

    with col2:

        st.metric(
            "Modelos evaluados",
            "7"
        )

    with col3:

        st.metric(
            "Modelo de simulación",
            "Regresión Lineal Bayesiana"
        )

    st.divider()

    st.success(
        "Sistema desarrollado para consumir modelos "
        "predictivos de crecimiento de clientes."
    )

    st.info(
        "La aplicación utiliza el modelo de "
        "Regresión Lineal Bayesiana actualizado "
        "para realizar la simulación."
    )

# ============================================================
# RESULTADOS
# ============================================================

elif menu == "📊 Resultados":

    st.title(
        "📊 Resultados de los Modelos"
    )

    st.markdown(
        """
        En esta sección se presentan los resultados obtenidos
        para los siete modelos de aprendizaje supervisado,
        utilizando los procedimientos de validación empleados
        en la investigación.
        """
    )

    # ========================================================
    # FUNCIÓN PARA MOSTRAR RESULTADOS
    # ========================================================

    def mostrar_resultados_modelos(df, titulo_validacion):

        if df is None or df.empty:

            st.warning(
                f"No existen resultados disponibles para "
                f"{titulo_validacion}."
            )

            return

        st.subheader(
            f"📋 Resultados - {titulo_validacion}"
        )

        # ----------------------------------------------------
        # MOSTRAR TABLA
        # ----------------------------------------------------

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        # ----------------------------------------------------
        # IDENTIFICAR COLUMNAS
        # ----------------------------------------------------

        columna_modelo = None
        columna_r2 = None
        columna_mse = None
        columna_mae = None

        for columna in df.columns:

            nombre = str(columna).upper()

            if nombre == "MODELO":
                columna_modelo = columna

            elif "R2" in nombre and "PROM" in nombre:
                columna_r2 = columna

            elif "MSE" in nombre and "PROM" in nombre:
                columna_mse = columna

            elif "MAE" in nombre and "PROM" in nombre:
                columna_mae = columna

        # ----------------------------------------------------
        # RESUMEN
        # ----------------------------------------------------

        st.subheader(
            "📌 Resumen de métricas"
        )

        resumen_cols = st.columns(3)

        # ----------------------------------------------------
        # MAYOR R²
        # ----------------------------------------------------

        if (
            columna_modelo is not None
            and columna_r2 is not None
        ):

            fila_r2 = df.loc[
                df[columna_r2].idxmax()
            ]

            with resumen_cols[0]:

                st.metric(
                    "Mayor R²",
                    normalizar_nombre_modelo(
                        fila_r2[columna_modelo]
                    )
                )

                st.caption(
                    f"R² = {fila_r2[columna_r2]:.4f}"
                )

        # ----------------------------------------------------
        # MENOR MSE
        # ----------------------------------------------------

        if (
            columna_modelo is not None
            and columna_mse is not None
        ):

            fila_mse = df.loc[
                df[columna_mse].idxmin()
            ]

            with resumen_cols[1]:

                st.metric(
                    "Menor MSE",
                    normalizar_nombre_modelo(
                        fila_mse[columna_modelo]
                    )
                )

                st.caption(
                    f"MSE = {fila_mse[columna_mse]:.4f}"
                )

        # ----------------------------------------------------
        # MENOR MAE
        # ----------------------------------------------------

        if (
            columna_modelo is not None
            and columna_mae is not None
        ):

            fila_mae = df.loc[
                df[columna_mae].idxmin()
            ]

            with resumen_cols[2]:

                st.metric(
                    "Menor MAE",
                    normalizar_nombre_modelo(
                        fila_mae[columna_modelo]
                    )
                )

                st.caption(
                    f"MAE = {fila_mae[columna_mae]:.4f}"
                )

        st.divider()

        # ====================================================
        # RANKING SEGÚN R²
        # ====================================================

        if (
            columna_modelo is not None
            and columna_r2 is not None
        ):

            st.subheader(
                "📈 Ranking de Modelos según R²"
            )

            df_r2 = df[
                [
                    columna_modelo,
                    columna_r2
                ]
            ].copy()

            df_r2["Modelo"] = (
                df_r2[columna_modelo]
                .apply(normalizar_nombre_modelo)
            )

            df_r2 = df_r2.sort_values(
                by=columna_r2,
                ascending=False
            )

            st.bar_chart(
                df_r2.set_index("Modelo")[columna_r2],
                use_container_width=True
            )

        st.divider()

        # ====================================================
        # RANKING SEGÚN MSE
        # ====================================================

        if (
            columna_modelo is not None
            and columna_mse is not None
        ):

            st.subheader(
                "📉 Ranking de Modelos según MSE"
            )

            df_mse = df[
                [
                    columna_modelo,
                    columna_mse
                ]
            ].copy()

            df_mse["Modelo"] = (
                df_mse[columna_modelo]
                .apply(normalizar_nombre_modelo)
            )

            df_mse = df_mse.sort_values(
                by=columna_mse,
                ascending=True
            )

            st.bar_chart(
                df_mse.set_index("Modelo")[columna_mse],
                use_container_width=True
            )

        st.divider()

        # ====================================================
        # RANKING SEGÚN MAE
        # ====================================================

        if (
            columna_modelo is not None
            and columna_mae is not None
        ):

            st.subheader(
                "📉 Ranking de Modelos según MAE"
            )

            df_mae = df[
                [
                    columna_modelo,
                    columna_mae
                ]
            ].copy()

            df_mae["Modelo"] = (
                df_mae[columna_modelo]
                .apply(normalizar_nombre_modelo)
            )

            df_mae = df_mae.sort_values(
                by=columna_mae,
                ascending=True
            )

            st.bar_chart(
                df_mae.set_index("Modelo")[columna_mae],
                use_container_width=True
            )


    # ========================================================
    # K-FOLD
    # ========================================================

    st.header(
        "🔄 Validación K-Fold"
    )

    df_kfold, archivo_kfold = cargar_resultados_kfold()

    if df_kfold is not None:

        st.caption(
            "Archivo de resultados: "
            + os.path.basename(archivo_kfold)
        )

        mostrar_resultados_modelos(
            df_kfold,
            "K-Fold"
        )

    else:

        st.warning(
            "No se encontró el archivo de resultados "
            "correspondiente a K-Fold."
        )


    # ========================================================
    # TIME SERIES SPLIT
    # ========================================================

    st.divider()

    st.header(
        "⏱️ Validación TimeSeriesSplit"
    )

    df_ts, archivo_ts = cargar_resultados_timeseries()

    if df_ts is not None:

        st.caption(
            "Archivo de resultados: "
            + os.path.basename(archivo_ts)
        )

        mostrar_resultados_modelos(
            df_ts,
            "TimeSeriesSplit"
        )

    else:

        st.warning(
            "No se encontró el archivo de resultados "
            "correspondiente a TimeSeriesSplit."
        )

# ============================================================
# SIMULACIÓN
# ============================================================

elif menu == "🔮 Simulación":

    st.title(
        "🔮 Simulación de Escenarios"
    )

    st.info(
        "Ingrese valores comerciales e históricos para "
        "estimar la cantidad de nuevos clientes del "
        "siguiente periodo."
    )

    st.markdown(
        f"""
        **Modelo utilizado: {NOMBRE_MODELO}**
        """
    )

    # ========================================================
    # VARIABLES COMERCIALES
    # ========================================================

    st.divider()

    st.subheader(
        "📋 Variables comerciales"
    )

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

    col1, col2, col3 = st.columns(3)

    with col1:

        programacion = st.selectbox(
            "Programación",
            [
                "BRONCE HD",
                "PLATA HD",
                "ORO HD"
            ]
        )

    with col2:

        mod_pago = st.selectbox(
            "Modalidad de pago",
            [
                "EFECTIVO",
                "TC"
            ]
        )

    with col3:

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

    st.divider()

    st.subheader(
        "📈 Historial reciente"
    )

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

        st.markdown(
            "### 📊 Promedio móvil"
        )

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

    # ========================================================
    # PREDICCIÓN
    # ========================================================

    st.divider()

    if st.button(
        "🚀 Generar Predicción",
        use_container_width=True
    ):

        try:

            # ------------------------------------------------
            # CARGAR MODELO BAYESIANO
            # ------------------------------------------------

            modelo = cargar_modelo()

            if modelo is None:

                st.error(
                    "No se encontró el archivo "
                    f"{MODELO_PRINCIPAL}."
                )

                st.stop()

            # ------------------------------------------------
            # CREAR ENTRADA
            # ------------------------------------------------

            entrada = preparar_entrada(
                decos=decos,
                mensualidad=mensualidad,
                programacion=programacion,
                mod_pago=mod_pago,
                estado=estado,
                lag1=lag1,
                lag3=lag3,
                lag6=lag6,
                rolling3=rolling3
            )

            # ------------------------------------------------
            # MOSTRAR ENTRADA
            # ------------------------------------------------

            with st.expander(
                "🔎 Ver datos utilizados para la predicción"
            ):

                st.dataframe(
                    entrada,
                    use_container_width=True,
                    hide_index=True
                )

            # ------------------------------------------------
            # GENERAR PREDICCIÓN
            # ------------------------------------------------

            prediccion = generar_prediccion(
                modelo,
                entrada
            )

            # ------------------------------------------------
            # EVITAR RESULTADOS NEGATIVOS
            # ------------------------------------------------

            prediccion_final = max(
                0,
                int(round(float(prediccion)))
            )

            # ------------------------------------------------
            # RESULTADO
            # ------------------------------------------------

            st.success(
                "Predicción generada correctamente."
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "👥 Clientes Predichos",
                    prediccion_final
                )

            with col2:

                st.metric(
                    "🤖 Modelo",
                    "Regresión Lineal Bayesiana"
                )

            st.info(
                "El resultado representa la cantidad "
                "estimada de nuevos clientes para el "
                "siguiente periodo."
            )

            st.write(
                "**Modelo utilizado:** "
                "Regresión Lineal Bayesiana"
            )

            st.write(
                "**Archivo del modelo:** "
                f"`{MODELO_PRINCIPAL}`"
            )

        except Exception as e:

            st.error(
                "Ocurrió un error al generar la predicción."
            )

            st.warning(
                "Verifique que las variables ingresadas "
                "tengan la misma estructura utilizada "
                "durante el entrenamiento del modelo."
            )

            st.exception(e)
