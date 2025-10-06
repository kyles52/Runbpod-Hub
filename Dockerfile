FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /

# System dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY builder/requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Pre-download models
RUN python3 -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
    print('Downloading FastVLM...'); \
    AutoTokenizer.from_pretrained('apple/FastVLM-0.5B', trust_remote_code=True); \
    AutoModelForCausalLM.from_pretrained('apple/FastVLM-0.5B', trust_remote_code=True, torch_dtype='float16')"

# Copy handler
COPY src/handler.py /handler.py

CMD ["python", "-u", "/handler.py"]