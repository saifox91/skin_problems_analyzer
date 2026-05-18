import gradio as gr
from fastai.vision.all import *
import platform
import pathlib

# Configuration pour Windows/Linux
if platform.system() == 'Linux': 
    pathlib.WindowsPath = pathlib.PosixPath

# Chargement du modèle
learn = load_learner('export.pkl')
labels = learn.dls.vocab

def predict(img):
    img = PILImage.create(img)
    _, pred_idx, probs = learn.predict(img)
    return {
        "predicted_class": labels[pred_idx],
        "all_predictions": [
            {"class": labels[i], "probability": float(probs[i])}
            for i in range(len(labels))
        ]
    }

# Interface API simple
app = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="filepath"),
    outputs=gr.JSON(),
    title="Skin Analyzer API",
    allow_flagging="never"
)

# Lancement avec accès public
app.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=True  # Crée un lien public
)