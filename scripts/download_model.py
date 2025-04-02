import os
import gdown

MODEL_DIR = "models"

MODEL_CONFIG = {
    "unet": {
        "filename": "061224_unet.keras",
        "file_id": "1hIzwIuIysYZewj8UBcp3tgzTXKFt91Qe"
    },
    "fcn": {
        "filename": "061224_fcn.keras",
        "file_id": "1ja5ZE41reHMp7qzAhgTXKiC-Rzu1WxhZ"
    }
}

def download_model(model_type="unet"):
    if model_type not in MODEL_CONFIG:
        raise ValueError("Invalid model type. Choose from: 'unet', 'fcn'")

    os.makedirs(MODEL_DIR, exist_ok=True)
    config = MODEL_CONFIG[model_type]
    output_path = os.path.join(MODEL_DIR, config["filename"])

    if os.path.exists(output_path):
        print(f"{model_type.upper()} model already exists.")
        return output_path

    url = f"https://drive.google.com/uc?id={config['file_id']}"
    print(f"⬇️  Downloading {model_type.upper()} model from {url} ...")
    gdown.download(url, output_path, quiet=False)
    print(f"Saved model to {output_path}")
    return output_path

if __name__ == "__main__":
    download_model("unet")
