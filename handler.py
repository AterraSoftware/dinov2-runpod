import torch
from torchvision import transforms
from PIL import Image
import requests
import io

# Chargement du modèle DINOv2
print("[INIT] Chargement du modèle...")
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
model.eval()
print("[INIT] Modèle chargé.")

# Prétraitement de l'image
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    ),
])

def handler(event):
    print("[HANDLER] Démarrage du handler...")
    try:
        # Log des inputs reçus
        print("[INPUT]", event)

        # Récupération de l'URL d'image
        image_url = event["input"]["image_url"]
        print(f"[DOWNLOAD] Téléchargement de l'image depuis {image_url}...")

        # Téléchargement et ouverture de l'image
        response = requests.get(image_url)
        image = Image.open(io.BytesIO(response.content)).convert("RGB")

        # Préparation de l'image
        input_tensor = preprocess(image).unsqueeze(0)

        # Inférence
        print("[INFERENCE] Génération du vecteur...")
        with torch.no_grad():
            output = model(input_tensor)

        # Extraction du vecteur
        vector = output.squeeze().tolist()

        print("[SUCCESS] Vecteur généré.")
        return {"vector": vector}

    except Exception as e:
        print("[ERROR]", str(e))
        return {"error": str(e)}
