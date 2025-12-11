import streamlit as st
import sqlite3
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download 
import os
# Importa a classe InputLayer para a correção
from tensorflow.keras.layers import InputLayer 

# --- Configurações do Modelo e Banco de Dados ---
REPO_ID = "viniciuslima47/pneumonia-vgg-model" 
MODEL_FILENAME = "final_vgg16_model.h5" 

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
    """Baixa o modelo do Hugging Face Hub e o carrega usando Keras com custom_objects."""
    
    st.info(f"Baixando e carregando o modelo do Hugging Face Hub ({MODEL_FILENAME})...")
    
    try:
        downloaded_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=MODEL_FILENAME
        )
        st.success("Download do modelo concluído. Carregando Keras...")

        # TENTA CORRIGIR O ERRO: Passa o InputLayer como objeto customizado,
        # forçando o Keras a aceitar o InputLayer que foi salvo de forma antiga.
        model = tf.keras.models.load_model(
            downloaded_path,
            custom_objects={'InputLayer': InputLayer}
        )
        return model
        
    except Exception as e:
        # Se o erro 'batch_shape' persistir, é porque a incompatibilidade é mais profunda.
        st.error(f"Erro ao carregar o modelo Keras. O formato .h5 está incompatível com as versões recentes do TensorFlow.")
        st.caption("Último erro tentando carregar o modelo .h5:")
        st.code(f"Detalhes do Erro: {e}")
        return None

# Carrega o modelo.
model = load_and_cache_model()

# --- Interface do Streamlit (Sem Alterações) ---
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
            arr = np.expand_dims(arr, axis=0)

            # Predição
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
    st.error("O modelo não pôde ser carregado. Verifique os logs para detalhes do erro.")
