FROM pytorch/pytorch:2.1.0-cuda11.8-devel

WORKDIR /app

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copie et installation des requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du handler
COPY handler.py .

# Pre-download du modèle pour réduire les cold starts
RUN python -c "from transformers import AutoImageProcessor, AutoModel; AutoImageProcessor.from_pretrained('facebook/dinov2-large'); AutoModel.from_pretrained('facebook/dinov2-large'); print('✅ Model cached')"

# Point d'entrée
CMD ["python", "handler.py"]
