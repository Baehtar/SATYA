import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="hf-inference",
    api_key=os.environ["HF_TOKEN"]
)

result = client.image_classification(
    "images.jpg",
    model="Organika/sdxl-detector"
)

print(result)