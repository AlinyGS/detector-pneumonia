import json
import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

AQUI = os.path.dirname(os.path.abspath(__file__))
CAMINHO_MODELO = os.path.join(AQUI, "modelo_pneumonia_v2.keras")
CAMINHO_LIMIAR = os.path.join(AQUI, "limiar.json")
# Saída sigmoide = probabilidade de PNEUMONIA (classes: NORMAL=0, PNEUMONIA=1).

st.set_page_config(page_title="Detector de Pneumonia", page_icon="🫁")


@st.cache_resource
def carregar():
    modelo = tf.keras.models.load_model(CAMINHO_MODELO)
    cfg = {"limiar": 0.5, "tamanho": 224}
    if os.path.exists(CAMINHO_LIMIAR):
        with open(CAMINHO_LIMIAR) as f:
            cfg.update(json.load(f))
    return modelo, cfg


def prever(modelo, tamanho, imagem: Image.Image) -> float:
    imagem = ImageOps.exif_transpose(imagem).convert("RGB").resize((tamanho, tamanho))
    x = np.asarray(imagem, dtype="float32")[None]  # o pré-processamento está dentro do modelo
    return float(modelo.predict(x, verbose=0)[0][0])


st.title("🫁 Detector de Pneumonia em Raio-X")
st.warning(
    "Projeto educacional. O resultado NÃO é diagnóstico médico — "
    "procure sempre um profissional de saúde."
)

if not os.path.exists(CAMINHO_MODELO):
    st.error("Modelo não encontrado. Rode treinar_com_minhas_imagens.py primeiro.")
    st.stop()

modelo, cfg = carregar()
limiar = cfg["limiar"]

aba_upload, aba_camera = st.tabs(["📁 Enviar arquivo", "📷 Usar câmera"])
imagem = None
with aba_upload:
    arquivo = st.file_uploader("Envie o raio-X", type=["jpg", "jpeg", "png", "bmp"])
    if arquivo:
        imagem = Image.open(arquivo)
with aba_camera:
    foto = st.camera_input("Aponte a câmera para o raio-X e tire a foto")
    if foto:
        imagem = Image.open(foto)

if imagem is not None:
    st.image(imagem, caption="Imagem analisada", use_container_width=True)
    with st.spinner("Analisando..."):
        p = prever(modelo, cfg["tamanho"], imagem)
    if p >= limiar:
        st.error(f"⚠️ Sinais de PNEUMONIA (probabilidade {p * 100:.1f}%)")
    else:
        st.success(f"✅ NORMAL (probabilidade de pneumonia {p * 100:.1f}%)")
    st.progress(float(p), text=f"Probabilidade de pneumonia: {p * 100:.1f}% (limiar de decisão: {limiar * 100:.0f}%)")
    if abs(p - limiar) < 0.15:
        st.info("Resultado próximo do limiar — incerto. Tente outra imagem, mais nítida.")
    st.caption("Fotos de tela/filme têm brilho e ângulo diferentes das imagens de treino; "
               "prefira o upload do arquivo original para maior precisão.")
