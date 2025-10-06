#!/usr/bin/env python3
"""
SOP Generator Serverless Handler
Processes egocentric video frames with FastVLM
"""

import runpod
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import base64
import cv2
import numpy as np
from PIL import Image
import io
import easyocr

# Initialize models (runs once on cold start, cached on warm starts)
print("🔥 Initializing models...")

# FastVLM
tokenizer = AutoTokenizer.from_pretrained(
    'apple/FastVLM-0.5B', 
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    'apple/FastVLM-0.5B',
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map="auto",
    low_cpu_mem_usage=True
)

# EasyOCR
ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

print(f"✅ Models loaded on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")


def process_frame(job):
    """
    Main handler for video frame processing
    
    Expected input:
    {
        "input": {
            "frame_base64": "base64_encoded_image",
            "task": "vlm" | "ocr" | "both",
            "prompt": "Optional VLM prompt"
        }
    }
    """
    try:
        job_input = job["input"]
        
        # Decode image
        frame_data = base64.b64decode(job_input["frame_base64"])
        image = Image.open(io.BytesIO(frame_data)).convert("RGB")
        
        result = {
            "status": "success",
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        }
        
        task = job_input.get("task", "both")
        
        # VLM Processing
        if task in ["vlm", "both"]:
            prompt = job_input.get("prompt", "Describe the actions and objects in this image")
            
            # Process with FastVLM
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            # Add your VLM processing logic here
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7
                )
            
            description = tokenizer.decode(outputs[0], skip_special_tokens=True)
            result["vlm_description"] = description
        
        # OCR Processing
        if task in ["ocr", "both"]:
            # Convert PIL to numpy for EasyOCR
            img_np = np.array(image)
            ocr_results = ocr_reader.readtext(img_np)
            
            result["ocr_text"] = [
                {
                    "bbox": bbox,
                    "text": text,
                    "confidence": conf
                }
                for bbox, text, conf in ocr_results
            ]
        
        return result
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "type": type(e).__name__
        }


# Start the serverless worker
if __name__ == "__main__":
    runpod.serverless.start({"handler": process_frame})