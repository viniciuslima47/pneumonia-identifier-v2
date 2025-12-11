import streamlit as st
import sqlite3
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download # Importação necessária para o Hugging Face
import os
# O 'requests' e 'io' não são mais necessários para este método

# --- Configurações do Modelo e Banco de Dados ---
# Seu repositório no Hugging Face (URL fornecida: https://huggingface.co/viniciuslima47/pneumonia-vgg-model)
REPO_ID = "viniciuslima47/pneumonia-vgg-model" 
MODEL_FILENAME = "final_vgg16_model.h5" # Presume que este é o nome do arquivo no seu repositório HF

# --- Configuração do Banco de Dados ---
conn = sqlite3.connect("usos.db")
cur = conn.cursor()
cur.execute("""CREATE TABLE IF NOT EXISTS historico (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                resultado TEXT,
                prob REAL,
                filename TEXT
              )""")
conn.commit()


# 1. Função para Baixar e Carregar o Modelo
@st.cache_resource
def load_and_cache_model():
    """Baixa o modelo do Hugging Face Hub e o carrega usando Keras."""
    
    st.info(f"Baixando e carregando o modelo do Hugging Face Hub ({MODEL_FILENAME})...")
    
    try:
        # Baixa o arquivo e armazena-o em cache localmente. 
        # Esta função cuida da verificação de existência e do download.
        downloaded_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=MODEL_FILENAME
        )
        st.success("Download do modelo concluído. Carregando Keras...")

        # Carregar o modelo Keras a partir do caminho baixado
        model = tf.keras.models.load_model(downloaded_path)
        return model
        
    except Exception as e:
        # Erro genérico de download/carregamento (inclui incompatibilidade Keras)
        st.error(f"Erro ao baixar ou carregar o modelo. Verifique o REPO_ID e o nome do arquivo.")
        st.code(f"Detalhes do Erro: {e}")
        return None

# Carrega o modelo.
model = load_and_cache_model()

# --- Interface do Streamlit ---
st.title("Classificador de Pneumonia em Raio-X")

if model is not None:
    uploaded = st.file_uploader("Envie uma imagem de Raio-X (peito)", type=["jpg","png","jpeg"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption='Imagem Enviada', use_column_width=True)
        st.divider()

        with st.spinner('Realizando a predição...'):
            # Pré-processamento
            img_resized = img.resize((224,224))
            arr = np.array(img_resized)/255.0
            arr = np.expand_dims(arr, axis=0) # Adiciona a dimensão do batch

            # Predição
            # Nota: O uso de model.predict pode gerar avisos, mas deve funcionar
            prob = float(model.predict(arr)[0][0])
            pred = "Pneumonia" if prob > 0.5 else "Normal"
            
            st.success("Predição concluída!")

        # Exibição dos resultados
        st.subheader(f"Diagnóstico: **{pred}**")
        st.metric(label="Probabilidade de Pneumonia (Classe 1)", value=f"{prob:.4f}")
        
        # Enviando para o banco de dados
        cur.execute("INSERT INTO historico (resultado, prob, filename) VALUES (?, ?, ?)",
                    (pred, prob, uploaded.name))
        conn.commit()
        
        # Mostrar Histórico
        if st.checkbox("Mostrar Histórico de Uso"):
            st.dataframe(cur.execute("SELECT * FROM historico ORDER BY id DESC").fetchall(),
                         column_names=["ID", "Resultado", "Prob.", "Nome do Arquivo"])

else:
    st.error("O modelo não pôde ser carregado.")
