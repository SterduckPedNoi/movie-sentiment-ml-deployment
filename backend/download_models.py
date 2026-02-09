import os
import zipfile
import gdown

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

MODELS = {
    "v6_transformer": "1ViTgT3aeQaME9NeXIJgzCMT1QfBV7D9h",
    "v7_albert": "1PEvge81aNMP2apqJCuNG-ReFbXpwaPva",
    "v8_distilroberta": "1yVzwkyAzE7ze5FJTyL1HI2qnmCyg9QFT",
    "v9_minilm": "1U7eEE3HFRKUfYZsDyos0udu-GQKtBq8K",
    "v10_minilm_l6": "1vDAiXvvyklwZ5HvFCXNxRgVetW7egDL8",
}

for name, file_id in MODELS.items():
    zip_path = os.path.join(MODELS_DIR, f"{name}.zip")
    model_path = os.path.join(MODELS_DIR, name)

    if os.path.exists(model_path):
        print(f"✅ {name} already exists, skip")
        continue

    print(f"⬇️ Downloading {name} ...")
    gdown.download(
        f"https://drive.google.com/uc?id={file_id}",
        zip_path,
        quiet=False,
        fuzzy=True
    )

    print(f"📦 Extracting {name} ...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(MODELS_DIR)

    os.remove(zip_path)
    print(f"🎉 {name} ready")
