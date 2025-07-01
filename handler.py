import runpod
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import io
import base64
import logging
import time

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Variables globales pour le modèle
processor = None
model = None
device = None

def init_model():
    """Initialisation du modèle DINOv2 - ViT-L/14 distilled"""
    global processor, model, device
    
    try:
        logger.info("🔄 Loading DINOv2 ViT-L/14 distilled model...")
        
        # Utilisation du modèle ViT-L/14 distilled (recommandé)
        model_name = 'facebook/dinov2-large'
        
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        logger.info(f"✅ Model loaded on {device}")
        logger.info(f"📊 Model: {model_name} (ViT-L/14 distilled)")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        return False

def extract_dinov2_features(image_data):
    """Extrait les features DINOv2 d'une image"""
    global processor, model, device
    
    try:
        start_time = time.time()
        
        # Décoder l'image base64
        if isinstance(image_data, str):
            # Enlever le préfixe data:image/... si présent
            if image_data.startswith('data:'):
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        else:
            image = image_data
        
        logger.info(f"📸 Image size: {image.size}")
        
        # Preprocessing DINOv2
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        # Extraction des features
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Token CLS (représentation globale de l'image)
        cls_token = outputs.last_hidden_state[:, 0, :]  # [1, 1024] pour ViT-L
        
        # Normalisation L2 pour la similarité cosinus
        normalized_embedding = F.normalize(cls_token, p=2, dim=1)
        
        # Conversion en liste Python
        embedding_vector = normalized_embedding.cpu().numpy().flatten().tolist()
        
        processing_time = time.time() - start_time
        logger.info(f"⚡ Feature extraction took {processing_time:.2f}s")
        
        return {
            "embedding": embedding_vector,
            "embedding_size": len(embedding_vector),
            "model_used": "dinov2-large (ViT-L/14 distilled)",
            "processing_time": processing_time
        }
        
    except Exception as e:
        logger.error(f"❌ Error extracting features: {str(e)}")
        raise

def classify_material_type(embedding):
    """Classification basique des matériaux basée sur l'embedding"""
    import numpy as np
    
    # Types de matériaux pour les textures architecturales
    materials = [
        "parquet", "carrelage", "beton", "pierre", 
        "marbre", "textile", "metal", "bois",
        "ceramique", "granit", "ardoise", "travertin"
    ]
    
    # Logique de classification simple (à améliorer avec un vrai classifieur)
    embedding_np = np.array(embedding)
    
    # Pour l'instant, classification aléatoire
    # Plus tard, vous pourrez entraîner un classifieur sur vos données
    material_scores = np.random.rand(len(materials))
    
    best_idx = np.argmax(material_scores)
    best_material = materials[best_idx]
    confidence = float(material_scores[best_idx])
    
    return {
        "detected_material": best_material,
        "confidence": confidence,
        "all_materials_scores": {
            mat: float(score) for mat, score in zip(materials, material_scores)
        }
    }

def handler(job):
    """Handler principal RunPod Serverless"""
    global processor, model, device
    
    try:
        logger.info("🚀 Starting DINOv2 feature extraction...")
        
        # Initialisation du modèle si pas encore fait
        if model is None:
            if not init_model():
                return {"error": "Failed to initialize model"}
        
        # Récupération des inputs
        job_input = job.get("input", {})
        
        # Vérification de la présence de l'image
        if "image" not in job_input:
            return {"error": "No 'image' field found in input"}
        
        # Extraction des features DINOv2
        features_result = extract_dinov2_features(job_input["image"])
        
        # Classification optionnelle du matériau
        material_classification = classify_material_type(features_result["embedding"])
        
        # Résultat final
        result = {
            "status": "success",
            "dinov2_features": features_result,
            "material_classification": material_classification,
            "timestamp": time.time()
        }
        
        logger.info("✅ Feature extraction completed successfully")
        logger.info(f"🎯 Detected material: {material_classification['detected_material']}")
        
        return result
        
    except Exception as e:
        error_msg = f"Handler error: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return {"error": error_msg, "status": "failed"}

# Point d'entrée RunPod
if __name__ == "__main__":
    logger.info("🚀 Starting DINOv2 RunPod Serverless Worker...")
    runpod.serverless.start({"handler": handler})
