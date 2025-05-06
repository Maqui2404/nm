import numpy as np
from PIL import Image
import cv2
import io
import streamlit as st

class ImagePreprocessor:
    """
    Clase para manejar el preprocesamiento de imágenes para el modelo de detección de neumonía.
    """
    
    def __init__(self, target_size=(150, 150)):
        """
        Inicializa el preprocesador de imágenes.
        
        Args:
            target_size (tuple): Tamaño objetivo de las imágenes (altura, ancho).
        """
        self.target_size = target_size
    
    def preprocess(self, image, normalize=True):
        """
        Preprocesa una imagen para que sea compatible con el modelo.
        
        Args:
            image (PIL.Image.Image): Imagen a preprocesar.
            normalize (bool): Si es True, normaliza los valores de píxeles al rango [0,1].
            
        Returns:
            numpy.ndarray: Imagen preprocesada como array numpy.
        """
        # Convertir a RGB si es necesario
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Redimensionar
        image = image.resize(self.target_size)
        
        # Convertir a array numpy
        img_array = np.array(image)
        
        # Normalizar si se especifica
        if normalize:
            img_array = img_array / 255.0
        
        # Expandir dimensiones para el batch
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def preprocess_file(self, file):
        """
        Preprocesa un archivo de imagen.
        
        Args:
            file: Objeto de archivo (como los proporcionados por st.file_uploader).
            
        Returns:
            tuple: (imagen original, imagen preprocesada como array numpy)
        """
        # Abrir la imagen
        image = Image.open(file)
        
        # Preprocesar
        img_array = self.preprocess(image)
        
        return image, img_array
    
    def preprocess_bytes(self, bytes_data):
        """
        Preprocesa datos de imagen en formato de bytes.
        
        Args:
            bytes_data (bytes): Datos de imagen en formato de bytes.
            
        Returns:
            tuple: (imagen original, imagen preprocesada como array numpy)
        """
        # Convertir bytes a imagen
        image = Image.open(io.BytesIO(bytes_data))
        
        # Preprocesar
        img_array = self.preprocess(image)
        
        return image, img_array
    
    def get_image_clahe(self, img_array, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization) para mejorar el contraste.
        
        Args:
            img_array (numpy.ndarray): Array de imagen.
            clip_limit (float): Límite de contraste para CLAHE.
            tile_grid_size (tuple): Tamaño de la cuadrícula para CLAHE.
            
        Returns:
            numpy.ndarray: Imagen con contraste mejorado.
        """
        # Eliminar la dimensión del batch y convertir a enteros de 8 bits
        if len(img_array.shape) == 4:
            img = (img_array[0] * 255).astype(np.uint8)
        else:
            img = (img_array * 255).astype(np.uint8)
        
        # Convertir a escala de grises si es RGB
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img
        
        # Aplicar CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        img_clahe = clahe.apply(img_gray)
        
        # Convertir de nuevo a RGB para visualización
        img_clahe_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)
        
        return img_clahe_rgb
    
    def get_enhanced_image(self, image):
        """
        Mejora una imagen aplicando varias técnicas de preprocesamiento.
        
        Args:
            image (PIL.Image.Image): Imagen a mejorar.
            
        Returns:
            PIL.Image.Image: Imagen mejorada.
        """
        # Convertir a array numpy
        img_array = np.array(image)
        
        # Aplicar CLAHE
        enhanced_array = self.get_image_clahe(img_array)
        
        # Convertir de nuevo a imagen PIL
        enhanced_image = Image.fromarray(enhanced_array)
        
        return enhanced_image

