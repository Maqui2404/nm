import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import cv2
import tensorflow as tf
from tensorflow.keras import Model
import base64
from io import BytesIO

class Visualizer:
    """
    Clase para crear visualizaciones relacionadas con el modelo y sus predicciones.
    """
    
    def __init__(self):
        """
        Inicializa el visualizador.
        """
        pass
    
    def plot_confidence_meter(self, value, class_name):
        """
        Crea un medidor visual para mostrar la confianza de la predicción.
        
        Args:
            value (float): Valor de confianza entre 0 y 1.
            class_name (str): Nombre de la clase predicha (NORMAL o PNEUMONIA).
            
        Returns:
            str: HTML para el medidor de confianza.
        """
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
        return html
    
    def plot_prediction_gauge(self, value):
        """
        Crea un gráfico de indicador para visualizar la probabilidad de neumonía.
        
        Args:
            value (float): Probabilidad de neumonía (entre 0 y 1).
            
        Returns:
            plotly.graph_objects.Figure: Figura de Plotly con el indicador.
        """
        # Crear figura con indicador de medidor
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = value * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Probabilidad de Neumonía (%)", 'font': {'size': 16}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "rgba(0,0,0,0)"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 50], 'color': '#d4edda'},
                    {'range': [50, 100], 'color': '#f8d7da'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        
        # Configurar layout
        fig.update_layout(
            height=250,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        
        return fig
    
    def plot_prediction_probabilities(self, normal_prob, pneumonia_prob):
        """
        Crea un gráfico de barras para visualizar las probabilidades de cada clase.
        
        Args:
            normal_prob (float): Probabilidad de la clase NORMAL.
            pneumonia_prob (float): Probabilidad de la clase PNEUMONIA.
            
        Returns:
            plotly.graph_objects.Figure: Figura de Plotly con el gráfico de barras.
        """
        fig = go.Figure()
        
        # Añadir barras
        fig.add_trace(go.Bar(
            x=['Normal', 'Neumonía'],
            y=[normal_prob, pneumonia_prob],
            marker_color=['#28a745', '#dc3545'],
            text=[f"{normal_prob:.2%}", f"{pneumonia_prob:.2%}"],
            textposition='auto'
        ))
        
        # Configurar layout
        fig.update_layout(
            title='Probabilidades por Clase',
            yaxis_title='Probabilidad',
            yaxis=dict(range=[0, 1], tickformat=".0%"),
            height=300,
            margin=dict(l=20, r=20, t=50, b=30),
            xaxis={'categoryorder':'total descending'},
        )
        
        return fig
    
    def plot_confusion_matrix(self, confusion_matrix=None):
        """
        Crea una visualización de la matriz de confusión.
        
        Args:
            confusion_matrix (numpy.ndarray, optional): Matriz de confusión personalizada. 
                Por defecto usa la del informe.
            
        Returns:
            plotly.graph_objects.Figure: Figura de Plotly con la matriz de confusión.
        """
        # Usar matriz de confusión por defecto del informe si no se proporciona
        if confusion_matrix is None:
            confusion_matrix = np.array([[366, 24], [24, 210]])
        
        # Crear figura
        fig = px.imshow(
            confusion_matrix,
            labels=dict(x="Predicción", y="Real", color="Conteo"),
            x=['PNEUMONIA', 'NORMAL'],
            y=['PNEUMONIA', 'NORMAL'],
            color_continuous_scale='blues',
            text_auto=True
        )
        
        # Personalizar layout
        fig.update_layout(
            title='Matriz de Confusión',
            xaxis_title='Predicción',
            yaxis_title='Real',
            width=400,
            height=400
        )
        
        return fig
    
    def generate_gradcam(self, model, img_array, layer_name='conv2d_4'):
        """
        Genera una visualización Grad-CAM (Class Activation Map) para la imagen.
        
        Args:
            model: Modelo TensorFlow/Keras.
            img_array (numpy.ndarray): Array de imagen preprocesada.
            layer_name (str): Nombre de la capa a usar para Grad-CAM.
            
        Returns:
            numpy.ndarray: Imagen Grad-CAM superpuesta sobre la original.
        """
        try:
            # Eliminar dimensión del batch si existe
            if len(img_array.shape) == 4:
                img = img_array[0]
            else:
                img = img_array
            
            # Asegurarse de que la imagen esté normalizada
            if img.max() > 1.0:
                img = img / 255.0
            
            # Crear modelo para obtener salidas de capa específica
            grad_model = Model(
                inputs=[model.inputs],
                outputs=[model.get_layer(layer_name).output, model.output]
            )
            
            # Grabiente cinta para calcular gradientes automáticamente
            with tf.GradientTape() as tape:
                # Tensor de entrada
                inputs = tf.cast(img_array, tf.float32)
                # Salidas de la capa específica y de la salida final
                conv_outputs, predictions = grad_model(inputs)
                # Como es clasificación binaria, tomamos directamente el valor
                # (si fuera multiclase, usaríamos argmax)
                class_idx = 0 if predictions[0][0] < 0.5 else 0
                loss = predictions[:, class_idx]
            
            # Calcular gradientes de la clase con respecto a la salida de la capa
            grads = tape.gradient(loss, conv_outputs)
            
            # Vector de importancia de canales: promedio espacial de los gradientes
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Multiplicar cada canal por su importancia y sumar
            conv_outputs = conv_outputs[0]
            heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
            
            # Normalizar el mapa de calor
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            heatmap = heatmap.numpy()
            
            # Redimensionar heatmap al tamaño de la imagen original
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            
            # Convertir a mapa de colores
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            # Imagen original a RGB y a uint8
            img_rgb = np.uint8(255 * img)
            
            # Superponer mapa de calor en la imagen original
            superimposed_img = cv2.addWeighted(img_rgb, 0.6, heatmap, 0.4, 0)
            
            return superimposed_img
            
        except Exception as e:
            st.error(f"Error generando Grad-CAM: {str(e)}")
            return None
    
    def get_image_download_link(self, img, filename="gradcam_result.jpg", text="Descargar imagen"):
        """
        Genera un enlace de descarga para una imagen.
        
        Args:
            img (numpy.ndarray): Imagen como array numpy.
            filename (str): Nombre del archivo a descargar.
            text (str): Texto para el enlace de descarga.
            
        Returns:
            str: HTML para el enlace de descarga.
        """
        buffered = BytesIO()
        if isinstance(img, np.ndarray):
            # Convertir de BGR a RGB si es necesario
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Guardar imagen en el buffer
            img_pil = Image.fromarray(img)
            img_pil.save(buffered, format="JPEG")
        else:
            # Si ya es una imagen PIL
            img.save(buffered, format="JPEG")
        
        # Codificar en base64
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Crear enlace de descarga HTML
        href = f'<a href="data:file/jpg;base64,{img_str}" download="{filename}">{text}</a>'
        
        return href
    
