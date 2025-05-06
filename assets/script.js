/**
 * Scripts personalizados para la aplicación de detección de neumonía
 */
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
// Función para añadir efecto de hover a las tarjetas
function setupCardEffects() {
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
}
// Función para animar el resultado de la predicción
function animatePredictionResult() {
    const resultElements = document.querySelectorAll('.result-container');
    
    resultElements.forEach(element => {
        element.style.opacity = '0';
        element.style.transform = 'translateY(20px)';
        
        setTimeout(() => {
            element.style.opacity = '1';
            element.style.transform = 'translateY(0)';
        }, 500);
    });
}

// Función para manejar la carga de imágenes
function handleImageUpload() {
    const fileInput = document.getElementById('image-upload');
    const previewContainer = document.getElementById('image-preview-container');
    const previewImage = document.getElementById('image-preview');
    const uploadLabel = document.querySelector('.upload-label');
    
    fileInput.addEventListener('change', function() {
        const file = this.files[0];
        if (file) {
            const reader = new FileReader();
            
            reader.addEventListener('load', function() {
                previewImage.setAttribute('src', this.result);
                previewContainer.classList.remove('hidden');
                uploadLabel.textContent = 'Cambiar imagen';
                
                // Reset de resultados anteriores si existen
                const resultsContainer = document.getElementById('results-container');
                if (resultsContainer) {
                    resultsContainer.classList.add('hidden');
                }
            });
            
            reader.readAsDataURL(file);
        }
    });
}

// Función para simular el proceso de análisis
function simulateAnalysis() {
    const analyzeBtn = document.getElementById('analyze-btn');
    const loadingIndicator = document.getElementById('loading-indicator');
    const resultsContainer = document.getElementById('results-container');
    
    analyzeBtn.addEventListener('click', function() {
        // Mostrar indicador de carga
        loadingIndicator.classList.remove('hidden');
        analyzeBtn.disabled = true;
        
        // Simular tiempo de procesamiento
        setTimeout(() => {
            // Ocultar indicador de carga
            loadingIndicator.classList.add('hidden');
            analyzeBtn.disabled = false;
            
            // Mostrar resultados
            resultsContainer.classList.remove('hidden');
            
            // Animar elementos de resultado
            animateConfidence();
            animatePredictionResult();
        }, 2000);
    });
}

// Función para inicializar tooltips
function initTooltips() {
    const tooltipTriggers = document.querySelectorAll('[data-tooltip]');
    
    tooltipTriggers.forEach(trigger => {
        trigger.addEventListener('mouseenter', () => {
            const tooltipText = trigger.getAttribute('data-tooltip');
            const tooltip = document.createElement('div');
            tooltip.className = 'tooltip';
            tooltip.textContent = tooltipText;
            
            document.body.appendChild(tooltip);
            
            const triggerRect = trigger.getBoundingClientRect();
            tooltip.style.top = `${triggerRect.bottom + 10}px`;
            tooltip.style.left = `${triggerRect.left + (triggerRect.width/2) - (tooltip.offsetWidth/2)}px`;
            tooltip.style.opacity = '1';
        });
        
        trigger.addEventListener('mouseleave', () => {
            const tooltip = document.querySelector('.tooltip');
            if (tooltip) {
                tooltip.remove();
            }
        });
    });
}

// Inicializar todas las funciones cuando el DOM esté cargado
document.addEventListener('DOMContentLoaded', function() {
    setupCardEffects();
    handleImageUpload();
    simulateAnalysis();
    initTooltips();
    
    // Navegación de pestañas si existe
    const tabLinks = document.querySelectorAll('.tab-link');
    if (tabLinks.length > 0) {
        tabLinks.forEach(link => {
            link.addEventListener('click', function(e) {
                e.preventDefault();
                
                // Eliminar clase activa de todos los enlaces
                tabLinks.forEach(l => l.classList.remove('active'));
                
                // Agregar clase activa al enlace actual
                this.classList.add('active');
                
                // Mostrar el contenido correspondiente
                const targetId = this.getAttribute('data-target');
                const tabContents = document.querySelectorAll('.tab-content');
                
                tabContents.forEach(content => {
                    content.classList.add('hidden');
                });
                
                document.getElementById(targetId).classList.remove('hidden');
            });
        });
    }
});