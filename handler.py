import torch
from torchvision import transforms
from PIL import Image
import requests
import io
import runpod

# Charger le modèle (DINOv2 ViT-L/14 Distilled)
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg', source='github')
model.eval()

# Préprocessing adapté à DINOv2
transform = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.CenterCrop(518),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def process_image_from_url(image_url):
    try:
        # Télécharger et préparer l'image
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            embedding = model(tensor)

        return {"vector": embedding.squeeze().tolist()}
    
    except requests.exceptions.RequestException as e:
        return {"error": f"Image download failed: {str(e)}"}
    except Exception as e:
        return {"error": f"Processing failed: {str(e)}"}

# Fonction principale pour Runpod
def handler(event):
    image_url = event.get("input", {}).get("image_url", None)
    if not image_url:
        return {"error": "No image URL provided."}

    return process_image_from_url(image_url)

# Lancer le worker
runpod.serverless.start({"handler": handler})
