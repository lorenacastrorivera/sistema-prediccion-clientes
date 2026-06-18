
import streamlit as st

st.set_page_config(
    page_title="Sistema Inteligente de Predicción",
    layout="wide"
)

st.title("📊 Sistema Inteligente de Predicción del Crecimiento de Clientes")

st.markdown("""
## Investigación

**Técnicas de aprendizaje supervisado para predecir el crecimiento de clientes de televisión por paga**

### Aplicabilidad

Esta herramienta puede adaptarse a:

- Empresas de telecomunicaciones
- Empresas de agua potable
- Empresas de energía eléctrica
- Empresas de internet
- Otros servicios

---
""")

col1, col2, col3 = st.columns(3)

col1.metric("Observaciones", "184")
col2.metric("Modelos", "7")
col3.metric("Mejor Modelo", "Bayesiana")

st.success(
    "Herramienta desarrollada para visualizar y consumir los modelos predictivos de la investigación."
)
