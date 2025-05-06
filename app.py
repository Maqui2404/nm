import os
import streamlit as st
from git import Repo

# Usar token de GitHub desde secrets
token = st.secrets["GITHUB_TOKEN"]
username = st.secrets["GITHUB_USERNAME"]

# Ruta del repo privado
repo_url = f"https://{username}:{token}@github.com/{username}/neu_mm.git"

# Ruta local para clonar
clone_dir = "/tmp/neu_mm"

# Clonar si no existe
if not os.path.exists(clone_dir):
    Repo.clone_from(repo_url, clone_dir)

# Ahora puedes acceder a tu modelo:
model_path = os.path.join(clone_dir, "modelo_neumonia.keras")

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import time
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import cv2
import base64

# Configuración de la página
st.set_page_config(
    page_title="Detector de Neumonía",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
def local_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    
    * {
        font-family: 'Roboto', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(to bottom, #f8f9fa, #e9ecef);
    }
    
    .title-container {
        background-color: #2c3e50;
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .title-container h1 {
        margin: 0;
        color: white;
        font-weight: 700;
    }
    
    .title-container p {
        margin-top: 0.5rem;
        opacity: 0.8;
    }
    
    .card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
    }
    
    .metrics-container {
        display: flex;
        justify-content: space-around;
        flex-wrap: wrap;
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        flex: 1;
        min-width: 150px;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.05);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0;
        color: #2c3e50;
    }
    
    .result-normal {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        margin: 1rem 0;
        animation: pulse 2s infinite;
    }
    
    .result-pneumonia {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        margin: 1rem 0;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% {
            box-shadow: 0 0 0 0 rgba(0, 123, 255, 0.4);
        }
        70% {
            box-shadow: 0 0 0 10px rgba(0, 123, 255, 0);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(0, 123, 255, 0);
        }
    }
    
    .upload-box {
        border: 2px dashed #adb5bd;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-box:hover {
        border-color: #0077b6;
        background-color: rgba(0, 119, 182, 0.05);
    }
    
    .stButton > button {
        background-color: #0077b6;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #023e8a;
        transform: scale(1.05);
    }
    
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #6c757d;
        font-size: 0.9rem;
    }
    
    /* Animación de carga */
    .loader {
        border: 5px solid #f3f3f3;
        border-top: 5px solid #3498db;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Animación para resultados */
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .confidence-meter {
        height: 20px;
        background-color: #e9ecef;
        border-radius: 10px;
        margin: 1rem 0;
        overflow: hidden;
        position: relative;
    }
    
    .confidence-value {
        height: 100%;
        border-radius: 10px;
        transition: width 1s ease-in-out;
    }
    </style>
    """, unsafe_allow_html=True)

# Inicializar la interfaz con el CSS
local_css()

# Función para añadir JavaScript para animaciones
def add_animation_js():
    js = """
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        // Función para animar el medidor de confianza
        function animateConfidence() {
            const meters = document.querySelectorAll('.confidence-value');
            meters.forEach(meter => {
                const targetWidth = meter.getAttribute('data-width');
                meter.style.width = '0%';
                setTimeout(() => {
                    meter.style.width = targetWidth + '%';
                }, 300);
            });
        }
        
        // Ejecutar animaciones
        animateConfidence();
        
        // Añadir efectos a las tarjetas
        const cards = document.querySelectorAll('.card');
        cards.forEach(card => {
            card.addEventListener('mouseenter', () => {
                card.style.transform = 'translateY(-10px)';
                card.style.boxShadow = '0 15px 25px rgba(0, 0, 0, 0.15)';
            });
            
            card.addEventListener('mouseleave', () => {
                card.style.transform = 'translateY(0)';
                card.style.boxShadow = '0 5px 15px rgba(0, 0, 0, 0.1)';
            });
        });
    });
    </script>
    """
    st.components.v1.html(js, height=0)

# Función para cargar y preparar el modelo
@st.cache_resource
def load_keras_model():
    try:
        model = load_model('modelo_neumonia.keras')
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

# Función para preprocesar la imagen
def preprocess_image(image, target_size=(150, 150)):
    # Convertir a escala de grises
    image = image.convert('L')  # 'L' = luminancia (escala de grises)
    
    # Redimensionar
    image = image.resize(target_size)
    
    # Convertir a array y normalizar
    img_array = np.array(image) / 255.0
    
    # Expandir dimensiones: (150,150) -> (1,150,150,1)
    img_array = np.expand_dims(img_array, axis=(0, -1))
    
    return img_array

# Función para predecir
def predict_pneumonia(model, img_array):
    prediction = model.predict(img_array)
    prediction_value = prediction[0][0]
    
    result = {}
    result['prediction'] = 'PNEUMONIA' if prediction_value > 0.5 else 'NORMAL'
    result['confidence'] = prediction_value if prediction_value > 0.5 else 1 - prediction_value
    result['pneumonia_prob'] = float(prediction_value)
    result['normal_prob'] = float(1 - prediction_value)
    
    return result

# Función para mostrar la matriz de confusión
def plot_confusion_matrix():
    # Datos de la matriz de confusión (del informe)
    cm = np.array([[366, 24], [24, 210]])
    
    # Crear la figura con plotly
    fig = px.imshow(
        cm,
        labels=dict(x="Predicción", y="Real", color="Conteo"),
        x=['PNEUMONIA', 'NORMAL'],
        y=['PNEUMONIA', 'NORMAL'],
        color_continuous_scale='blues',
        text_auto=True
    )
    
    fig.update_layout(
        title='Matriz de Confusión',
        xaxis_title='Predicción',
        yaxis_title='Real',
        width=400,
        height=400
    )
    
    return fig

# Función para visualizar métricas del modelo
def display_metrics():
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div>Precisión</div>
            <div class="metric-value">92.31%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div>Precisión Neumonía</div>
            <div class="metric-value">94%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div>Recall Neumonía</div>
            <div class="metric-value">94%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div>F1-Score</div>
            <div class="metric-value">94%</div>
        </div>
        """, unsafe_allow_html=True)

# Componente de medidor de confianza animado
def confidence_meter(value, class_name):
    color = "#28a745" if class_name == "NORMAL" else "#dc3545"
    
    html = f"""
    <div style="margin: 1.5rem 0;">
        <h4 style="margin-bottom: 0.5rem;">Nivel de Confianza: {value:.2%}</h4>
        <div class="confidence-meter">
            <div class="confidence-value" 
                 style="background-color: {color}; width: {value*100}%;"
                 data-width="{value*100}">
            </div>
        </div>
    </div>
    """
    return st.markdown(html, unsafe_allow_html=True)

# Función principal que construye la interfaz
def main():
    # Añadir JavaScript para animaciones
    add_animation_js()
    
    # Título de la aplicación
    st.markdown("""
    <div class="title-container">
        <h1>Detector de Neumonía en Radiografías</h1>
        <p>Aplicación de demostración para clasificación de neumonía mediante IA</p>
                <p>Tesista Marco Mayta</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Crear pestañas
    tab1, tab2, tab3 = st.tabs(["Diagnóstico", "Información del Modelo", "Ayuda"])
    
    with tab1:
        st.markdown("""
        <div class="card">
            <h2>Subir Radiografía</h2>
            <p>Suba una imagen de rayos X de tórax para detectar posibles signos de neumonía.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Área para subir la imagen
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Seleccione o arrastre una imagen de rayos X", type=["jpg", "jpeg", "png"])
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            # Cargar el modelo
            model = load_keras_model()
            
            if model:
                # Mostrar la imagen
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.subheader("Imagen Cargada")
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Radiografía de Tórax", use_column_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.subheader("Resultados del Análisis")
                    
                    # Procesar la imagen y hacer la predicción
                    with st.spinner("Analizando imagen..."):
                        # Simular tiempo de procesamiento para mostrar el spinner
                        time.sleep(1)
                        img_array = preprocess_image(image)
                        result = predict_pneumonia(model, img_array)
                    
                    # Mostrar resultado
                    if result['prediction'] == 'PNEUMONIA':
                        st.markdown(f"""
                        <div class="result-pneumonia fade-in">
                            NEUMONÍA DETECTADA
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-normal fade-in">
                            NORMAL - NO SE DETECTÓ NEUMONÍA
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Mostrar medidor de confianza
                    confidence_meter(result['confidence'], result['prediction'])
                    
                    # Mostrar probabilidades
                    st.markdown("""
                    <div style="margin-top: 1.5rem;">
                        <h4>Probabilidades:</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Crear un gráfico de barras para las probabilidades
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=['Normal', 'Neumonía'],
                        y=[result['normal_prob'], result['pneumonia_prob']],
                        marker_color=['#28a745', '#dc3545']
                    ))
                    fig.update_layout(
                        title='Probabilidades por Clase',
                        yaxis_title='Probabilidad',
                        yaxis=dict(range=[0, 1]),
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("**Nota:** Esta herramienta es solo para fines demostrativos y no sustituye el diagnóstico médico profesional.")
                    st.markdown('</div>', unsafe_allow_html=True)
        
        else:
            # Mensaje cuando no hay imagen cargada
            st.info("Por favor, suba una imagen de rayos X para comenzar el análisis.")
    
    with tab2:
        # Información sobre el modelo
        st.markdown("""
        <div class="card">
            <h2>Información del Modelo</h2>
            <p>Este modelo de inteligencia artificial ha sido entrenado para detectar signos de neumonía en radiografías de tórax.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Métricas del Modelo")
            display_metrics()
            
            st.subheader("Matriz de Confusión")
            confusion_fig = plot_confusion_matrix()
            st.plotly_chart(confusion_fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Detalles del Conjunto de Datos")
            st.markdown("""
            **Cantidad de datos por clase:**
            - Entrenamiento: PNEUMONIA: 3875, NORMAL: 1341
            - Validación: PNEUMONIA: 8, NORMAL: 8
            - Prueba: PNEUMONIA: 390, NORMAL: 234
            
            Este modelo fue entrenado con una arquitectura CNN que incluye:
            - 5 capas convolucionales
            - Normalización por lotes
            - Dropout para regularización
            - 2 capas densas
            
            Total de parámetros: 2,491,716 (9.51 MB)
            """)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        # Sección de ayuda
        st.markdown("""
        <div class="card">
            <h2>Ayuda e Información</h2>
            <p>Cómo utilizar esta aplicación y entender los resultados.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Cómo usar la aplicación")
        st.markdown("""
        1. Ve a la pestaña **Diagnóstico**
        2. Sube una imagen de rayos X de tórax (formatos JPG, JPEG o PNG)
        3. Espera a que el modelo analice la imagen
        4. Revisa los resultados y la confianza de la predicción
        
        **Importante:** Esta aplicación es solo una demostración y no debe utilizarse para diagnósticos médicos reales. Siempre consulta a un profesional de la salud.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Entendiendo los resultados")
        st.markdown("""
        - **Normal:** La IA no detectó signos de neumonía en la radiografía.
        - **Neumonía:** La IA detectó patrones consistentes con neumonía.
        
        El **nivel de confianza** indica qué tan seguro está el modelo de su predicción. Un valor más alto indica mayor confianza.
        
        La **matriz de confusión** muestra:
        - Verdaderos Positivos: 366 casos de neumonía correctamente identificados
        - Falsos Negativos: 24 casos de neumonía clasificados incorrectamente como normales
        - Falsos Positivos: 24 casos normales clasificados incorrectamente como neumonía
        - Verdaderos Negativos: 210 casos normales correctamente identificados
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Pie de página
    st.markdown("""
    <div class="footer">
        <p>Aplicación de demostración para detección de neumonía | 2025</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
