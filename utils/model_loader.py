import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import os
import numpy as np
import time

class ModelLoader:
    """
    Clase para manejar la carga y operaciones del modelo de detección de neumonía.
    """
    
    def __init__(self, model_path="modelo_neumonia.keras"):
        """
        Inicializa el cargador de modelos.
        
        Args:
            model_path (str): Ruta al archivo del modelo Keras.
        """
        self.model_path = model_path
        self.model = None
        self.input_shape = (150, 150, 3)  # Forma esperada de entrada para el modelo
    
    @st.cache_resource
    def load(self):
        """
        Carga el modelo desde el archivo.
        Utiliza st.cache_resource para evitar recargar el modelo en cada interacción.
        
        Returns:
            El modelo cargado o None si hay un error.
        """
        try:
            # Verificar si el archivo existe
            if not os.path.exists(self.model_path):
                st.error(f"No se encontró el archivo del modelo en {self.model_path}")
                return None
            
            # Cargar el modelo
            start_time = time.time()
            model = load_model(self.model_path)
            load_time = time.time() - start_time
            
            # Registrar información del modelo
            st.session_state['model_load_time'] = f"{load_time:.2f} segundos"
            
            # Verificar que el modelo tenga la forma de entrada esperada
            input_shape = model.input_shape[1:]
            if input_shape != self.input_shape:
                st.warning(
                    f"Advertencia: La forma de entrada del modelo cargado {input_shape} "
                    f"no coincide con la esperada {self.input_shape}. "
                    f"Ajustando preprocesamiento..."
                )
            
            self.model = model
            return model
            
        except Exception as e:
            st.error(f"Error al cargar el modelo: {str(e)}")
            return None
    
    def predict(self, img_array):
        """
        Realiza una predicción utilizando el modelo cargado.
        
        Args:
            img_array (numpy.ndarray): Array preprocesado de la imagen.
            
        Returns:
            dict: Diccionario con los resultados de la predicción.
        """
        if self.model is None:
            self.model = self.load()
            if self.model is None:
                return {"error": "No se pudo cargar el modelo"}
        
        try:
            # Verificar forma de la imagen
            if len(img_array.shape) != 4:
                return {"error": "La imagen debe tener forma (batch, height, width, channels)"}
            
            # Hacer la predicción
            start_time = time.time()
            prediction = self.model.predict(img_array)
            inference_time = time.time() - start_time
            
            # Procesar resultados
            prediction_value = prediction[0][0]
            
            result = {
                "prediction": "PNEUMONIA" if prediction_value > 0.5 else "NORMAL",
                "confidence": prediction_value if prediction_value > 0.5 else 1 - prediction_value,
                "pneumonia_prob": float(prediction_value),
                "normal_prob": float(1 - prediction_value),
                "inference_time": f"{inference_time:.4f} segundos"
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Error durante la predicción: {str(e)}"}
    
    def get_model_summary(self):
        """
        Obtiene un resumen del modelo en formato de texto.
        
        Returns:
            str: Resumen del modelo en formato de texto.
        """
        if self.model is None:
            self.model = self.load()
            if self.model is None:
                return "No se pudo cargar el modelo para obtener el resumen."
        
        # Capturar el resumen del modelo
        summary_lines = []
        self.model.summary(print_fn=lambda x: summary_lines.append(x))
        
        return "\n".join(summary_lines)
    