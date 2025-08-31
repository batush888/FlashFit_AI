import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from typing import Union, List, Optional
import numpy as np

class BLIPCaptioner:
    """BLIP captioner for generating rich natural language descriptions of garments"""
    
    def __init__(self, model_id="Salesforce/blip-image-captioning-base", device=None):
        """
        Initialize BLIP captioner
        
        Args:
            model_id: HuggingFace model identifier
            device: Device to run on (auto-detect if None)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        
        # Load processor and model
        self.processor = BlipProcessor.from_pretrained(model_id)
        self.model = BlipForConditionalGeneration.from_pretrained(model_id).to(self.device)
        self.model.eval()
        
        print(f"BLIPCaptioner initialized with {model_id} on {self.device}")
    
    def caption(self, img_path: Union[str, Image.Image], max_new_tokens: int = 30) -> str:
        """
        Generate caption for an image
        
        Args:
            img_path: Path to image file or PIL Image object
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated caption as string
        """
        # Handle different input types
        if isinstance(img_path, str):
            raw_image = Image.open(img_path).convert("RGB")
        elif isinstance(img_path, Image.Image):
            raw_image = img_path.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(img_path)}")
        
        # Process image and generate caption
        inputs = self.processor(raw_image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption
    
    def conditional_caption(self, img_path: Union[str, Image.Image], 
                          prompt: str, max_new_tokens: int = 30) -> str:
        """
        Generate conditional caption with a text prompt
        
        Args:
            img_path: Path to image file or PIL Image object
            prompt: Text prompt to condition the caption
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated conditional caption as string
        """
        # Handle different input types
        if isinstance(img_path, str):
            raw_image = Image.open(img_path).convert("RGB")
        elif isinstance(img_path, Image.Image):
            raw_image = img_path.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(img_path)}")
        
        # Process image with text prompt
        inputs = self.processor(raw_image, prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption
    
    def batch_caption(self, image_paths: List[Union[str, Image.Image]], 
                     max_new_tokens: int = 30) -> List[str]:
        """
        Generate captions for multiple images
        
        Args:
            image_paths: List of image paths or PIL Image objects
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            List of generated captions
        """
        captions = []
        
        for img_path in image_paths:
            caption = self.caption(img_path, max_new_tokens)
            captions.append(caption)
        
        return captions
    
    def generate_fashion_description(self, img_path: Union[str, Image.Image]) -> dict:
        """
        Generate comprehensive fashion description using multiple prompts
        
        Args:
            img_path: Path to image file or PIL Image object
            
        Returns:
            Dictionary with different types of descriptions
        """
        # Different prompts for fashion analysis
        prompts = {
            "general": "",  # No prompt for general caption
            "style": "describe the style of this clothing item:",
            "color": "what colors are in this clothing item:",
            "material": "what material is this clothing made of:",
            "occasion": "what occasion is this clothing suitable for:"
        }
        
        descriptions = {}
        
        # General caption without prompt
        descriptions["general"] = self.caption(img_path)
        
        # Conditional captions with specific prompts
        for key, prompt in prompts.items():
            if key != "general" and prompt:
                descriptions[key] = self.conditional_caption(img_path, prompt)
        
        return descriptions
    
    def extract_attributes(self, img_path: Union[str, Image.Image]) -> List[str]:
        """
        Extract fashion attributes from image using targeted prompts
        
        Args:
            img_path: Path to image file or PIL Image object
            
        Returns:
            List of extracted attributes
        """
        attribute_prompts = [
            "list the clothing type:",
            "describe the pattern:",
            "identify the fit:",
            "describe the neckline:",
            "identify the sleeve type:"
        ]
        
        attributes = []
        
        for prompt in attribute_prompts:
            try:
                attr = self.conditional_caption(img_path, prompt, max_new_tokens=20)
                # Clean up the attribute (remove prompt repetition)
                attr_clean = attr.replace(prompt, "").strip()
                if attr_clean:
                    attributes.append(attr_clean)
            except Exception as e:
                print(f"Error extracting attribute with prompt '{prompt}': {e}")
                continue
        
        return attributes
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """
        Get text embedding from BLIP model (for similarity computation)
        
        Args:
            text: Input text
            
        Returns:
            Text embedding as numpy array
        """
        # For BLIP captioning model, we'll use the text encoder part
        # Tokenize text
        inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Get text embeddings from the text encoder
            text_outputs = self.model.text_encoder(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            # Use the pooled output (CLS token representation)
            text_features = text_outputs.pooler_output
            # Normalize
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy()

# Global instance for reuse
_blip_captioner = None

def get_blip_captioner() -> BLIPCaptioner:
    """Get global BLIP captioner instance"""
    global _blip_captioner
    if _blip_captioner is None:
        _blip_captioner = BLIPCaptioner()
    return _blip_captioner