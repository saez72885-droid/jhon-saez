import os
import time
import streamlit as st
import chromadb
from dotenv import load_dotenv
from google import genai

# 1. Cargamos tu clave secreta desde el archivo .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Configuración visual pensada para celular y escritorio
st.set_page_config(
    page_title="Asistente Normativo RIC Chile",
    page_icon="⚡",
    layout="centered"
)

st.title("⚡ Asistente Normativo RIC")
st.caption("Consulta inteligente sobre los 19 Pliegos Técnicos Normativos (SEC - Chile)")

if not api_key:
    st.error("No se encontró GEMINI_API_KEY en el archivo .env.")
    st.stop()

# 2. Conectamos con la base de datos vectorial ya creada
@st.cache_resource
def cargar_base_datos():
    cliente_db = chromadb.PersistentClient(path="base_ric")
    return cliente_db.get_collection(name="pliegos_ric")

coleccion = cargar_base_datos()
cliente_ia = genai.Client(api_key=api_key)

# 3. Filtro opcional por Pliego en el menú lateral
lista_pliegos = [
    "Todos los RIC",
    "RIC-N01-Empalmes",
    "RIC-N02-Tableros-Electricos",
    "RIC-N03-Alimentadores-y-demanda-de-una-instalacion-V1.1-1",
    "RIC-N04-Conductores-y-Canalizaciones",
    "RIC-N05-Medidas-de-Proteccion-Contra-Tensiones-Peligrosas",
    "RIC-N06-Puesta-a-Tierra",
    "RIC-N07-Instalaciones-de-Equipos-V1.1-1",
    "RIC-N08-Sistemas-de-Emergencia",
    "RIC-N09-Sistemas-de-autogeneracion",
    "RIC-N10-Instalaciones-de-uso-general",
    "RIC-N11-Instalaciones-Especiales",
    "RIC-N12-Instalaciones-en-Ambientes-Explosivos",
    "RIC-N13-Subestaciones-y-Salas-Electricas",
    "RIC-N14-Exigencias-de-eficiencia-energetica-para-edificios",
    "RIC-N15-Version-2024",
    "RIC-N16-Subsistemas-de-Distribución",
    "RIC-N17-Operacion-y-Mantenimiento",
    "RIC-N18-Presentacion-de-Proyectos",
    "RIC-N19-Puesta-en-Servicio"
]

with st.sidebar:
    st.header("Filtros de búsqueda")
    pliego_seleccionado = st.selectbox("Buscar en:", lista_pliegos)
    num_resultados = st.slider("Profundidad de búsqueda (fragmentos)", 4, 12, 7)

# 4. Historial de conversación en pantalla
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

for msg in st.session_state.mensajes:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 5. Caja de pregunta del usuario
pregunta = st.chat_input("Ej: ¿Qué sección mínima y diferencial exige la norma para enchufes en cocina?")

if pregunta:
    st.session_state.mensajes.append({"role": "user", "content": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)

    with st.chat_message("assistant"):
        with st.spinner("Buscando en los Pliegos RIC..."):
            if pliego_seleccionado == "Todos los RIC":
                resultados = coleccion.query(
                    query_texts=[pregunta],
                    n_results=num_resultados
                )
            else:
                resultados = coleccion.query(
                    query_texts=[pregunta],
                    n_results=num_resultados,
                    where={"pliego": pliego_seleccionado}
                )

            fragmentos = resultados["documents"][0]
            metadatos = resultados["metadatas"][0]
            contexto_normativo = "\n\n---\n\n".join(fragmentos)

            instrucciones = f"""
            Eres un ingeniero eléctrico experto en la normativa eléctrica chilena vigente (Pliegos Técnicos Normativos RIC del DS N° 08 de la SEC).
            Responde la consulta del usuario basándote ÚNICA Y ESTRICTAMENTE en los siguientes extractos oficiales recuperados de los RIC:

            {contexto_normativo}

            Reglas obligatorias:
            1. Cita siempre el Pliego RIC exacto, el número de punto/cláusula, tabla y página de donde sacas cada dato.
            2. Si hay valores técnicos (secciones en mm², capacidades en Amperes, tiempos de disparo en ms, factores de corrección, alturas, etc.), entrégalos con total precisión.
            3. Si la información entregada no alcanza para responder con certeza, dilo claramente sin inventar datos.
            4. Estructura la respuesta de forma clara y práctica para alguien que está en terreno o diseñando un proyecto.

            Consulta del usuario: {pregunta}
            """

            # Obtenemos automáticamente los modelos Gemini reales disponibles en tu cuenta
            modelos_disponibles = [
                m.name.replace("models/", "")
                for m in cliente_ia.models.list()
                if "gemini" in m.name.lower() and "flash" in m.name.lower()
            ]
            if not modelos_disponibles:
                modelos_disponibles = ["gemini-3.8-flash"]

            respuesta = None
            for intento in range(3):
                for modelo in modelos_disponibles:
                    try:
                        respuesta = cliente_ia.models.generate_content(
                            model=modelo,
                            contents=instrucciones
                        )
                        break
                    except Exception as e:
                        err = str(e)
                        if "503" in err or "UNAVAILABLE" in err or "404" in err or "429" in err:
                            continue
                        raise e
                if respuesta is not None:
                    break
                time.sleep(2)

            if respuesta is None:
                st.warning("Los servidores de Google están con alta demanda en este segundo. Intenta de nuevo en unos segundos.")
                st.stop()

            texto_respuesta = respuesta.text
            st.markdown(texto_respuesta)

            with st.expander("📄 Ver fragmentos originales consultados"):
                for meta, frag in zip(metadatos, fragmentos):
                    st.markdown(f"**{meta['pliego']} — Página {meta['pagina']}**")
                    st.text(frag)
                    st.divider()

            st.session_state.mensajes.append({"role": "assistant", "content": texto_respuesta})