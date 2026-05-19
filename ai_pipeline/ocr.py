# from huggingface_hub import hf_hub_download

# # Download English models
# det_path = hf_hub_download("monkt/paddleocr-onnx", "detection/v5/det.onnx")
# rec_path = hf_hub_download("monkt/paddleocr-onnx", "languages/eslav/rec.onnx")
# dict_path = hf_hub_download("monkt/paddleocr-onnx", "languages/eslav/dict.txt")
from rapidocr_onnxruntime import RapidOCR

ocr = RapidOCR(
    det_model_path="det.onnx",
    rec_model_path="rec.onnx",
    rec_keys_path="dict.txt"
)

result, elapsed = ocr("image.png")
print(result)

print(elapsed)

