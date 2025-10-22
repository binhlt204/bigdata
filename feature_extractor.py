
import torch
from PIL import Image
from conch.open_clip_custom.factory import create_model_from_pretrained

Image.MAX_IMAGE_PIXELS = None

class FeatureExtractor:
    def __init__(self, ckpt_path="ckpts/conch.pth", device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        model_cfg = 'conch_ViT-B-16'
        self.model, self.preprocess = create_model_from_pretrained(
            model_cfg, ckpt_path, device=self.device
        )
        self.model.eval()

    def extract_features(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
            tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.model.encode_image(tensor, return_tokens=True)
                
                return feat.squeeze(0).cpu().numpy().tolist()
        except Exception as e:
            print(f"⚠️ Error {image_path}: {str(e)}")
            return None
    
