import torch
from PIL import Image
from transformers import SiglipProcessor, SiglipTextModel, SiglipVisionModel
import logging
import warnings

# Suppress transformers verbose warnings/logs
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)

class SiglipEmbedder:
    def __init__(self, model_name: str = "google/siglip-base-patch16-224"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = None
        self._text_model = None
        self._vision_model = None

    @property
    def processor(self) -> SiglipProcessor:
        if self._processor is None:
            # Try offline loading first (skips HuggingFace Hub network checks)
            try:
                self._processor = SiglipProcessor.from_pretrained(
                    self.model_name, 
                    local_files_only=True
                )
            except Exception:
                # Fallback to online loading if cache is empty
                self._processor = SiglipProcessor.from_pretrained(self.model_name)
        return self._processor

    @property
    def text_model(self) -> SiglipTextModel:
        if self._text_model is None:
            try:
                self._text_model = SiglipTextModel.from_pretrained(
                    self.model_name, 
                    local_files_only=True
                ).to(self.device)
            except Exception:
                self._text_model = SiglipTextModel.from_pretrained(self.model_name).to(self.device)
            self._text_model.eval()
        return self._text_model

    @property
    def vision_model(self) -> SiglipVisionModel:
        if self._vision_model is None:
            try:
                self._vision_model = SiglipVisionModel.from_pretrained(
                    self.model_name, 
                    local_files_only=True
                ).to(self.device)
            except Exception:
                self._vision_model = SiglipVisionModel.from_pretrained(self.model_name).to(self.device)
            self._vision_model.eval()
        return self._vision_model

    def get_image_embedding(self, image: Image.Image) -> list[float]:
        """Generate normalized embedding for a PIL Image."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.vision_model(**inputs)
            image_features = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
            # L2 normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features[0].cpu().numpy().tolist()

    def get_text_embedding(self, text: str) -> list[float]:
        """Generate normalized embedding for a text query."""
        inputs = self.processor(text=[text], padding="max_length", return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.text_model(**inputs)
            text_features = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
            # L2 normalize
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features[0].cpu().numpy().tolist()

