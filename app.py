import streamlit as st
import pandas as pd
import joblib


# ========================================================
# CONFIGURACIÓN
# ========================================================

st.set_page_config(
    page_title="Predicción de Nuevos Clientes",
    page_icon="📊",
    layout="wide"
)


# ========================================================
# TÍTULO
# ========================================================

st.title(
    "📊 Predicción del crecimiento de nuevos clientes"
)

st.write(
    "Aplicación de predicción basada en técnicas "
    "de aprendizaje supervisado."
)


st.divider()


# ========================================================
# INFORMACIÓN COMERCIAL
# ========================================================

st.subheader(
    "📋 Características del servicio"
)


col1, col2 = st.columns(2)


with col1:

    decos = st.number_input(
        "Decos promedio",
        min_value=0.0,
        value=1.0,
        step=1.0
    )


    mensualidad = st.number_input(
        "Mensualidad promedio",
        min_value=0.0,
        value=140.0,
        step=1.0
    )


with col2:

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

st.subheader(
    "📈 Historial reciente"
)


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


# ========================================================
# PROMEDIO MÓVIL
# ========================================================

rolling3 = (
    lag1 +
    lag2 +
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
        # CARGAR MODELO DE REGRESIÓN CUANTIL
        # ------------------------------------------------

        modelo = joblib.load(
            "modelo_regresion_cuantil.pkl"
        )


        # ------------------------------------------------
        # CREAR DATAFRAME DE ENTRADA
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
        # PREDICCIÓN
        # ------------------------------------------------

        prediccion = modelo.predict(
            entrada
        )[0]


        # ------------------------------------------------
        # CONVERTIR A CANTIDAD ENTERA
        # ------------------------------------------------

        prediccion_final = max(
            0,
            int(round(prediccion))
        )


        # ------------------------------------------------
        # RESULTADO
        # ------------------------------------------------

        st.success(
            "Predicción generada correctamente."
        )


        st.metric(
            "👥 Clientes Predichos",
            prediccion_final
        )


        st.info(
            "El resultado representa la cantidad "
            "estimada de nuevos clientes para "
            "el siguiente periodo."
        )


        st.write(
            "**Modelo utilizado:** "
            "Regresión Cuantil"
        )


        st.write(
            "**Cuantil:** 0.50"
        )


    except FileNotFoundError:

        st.error(
            "No se encontró el archivo "
            "'modelo_regresion_cuantil.pkl'. "
            "Verifique que se encuentre en la "
            "misma carpeta que la aplicación."
        )


    except Exception as e:

        st.error(
            "Ocurrió un error al generar "
            "la predicción."
        )

        st.exception(e)
