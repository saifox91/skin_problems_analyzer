import gradio as gr
from fastai.vision.all import *
import platform
import pathlib
import traceback # Importez ceci

# Correction des chemins pour Windows sur Linux
if platform.system() == 'Linux':
    pathlib.WindowsPath = pathlib.PosixPath

# Charger le modèle
learn = load_learner('export.pkl')
labels = learn.dls.vocab

def predict(img): # img est le chemin du fichier temporaire
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
    inputs=gr.Image(type="filepath"), # Ici, 'filepath' est le bon type pour le nom de fichier temporaire
    outputs=gr.JSON(),
    title="Skin Analyzer API",
    allow_flagging="never"
)

# Lancement avec accès public
app.launch(
    server_name="0.0.0.0",
    server_port=7860,
    # share=True  # Assurez-vous que cette ligne est bien commentée ou supprimée !
)

# ====================================================================
# IMPORTANT : Testez la fonction predict_api directement via l'interface web Gradio si possible
# Normalement, votre `app` (qui est un gr.Interface) expose déjà /api/predict
#
# Voici comment vous pouvez appeler cette API via votre navigateur (outil POSTMAN ou CURL) :
# URL: https://saifox-skin-problems-analyzer.hf.space/api/predict
# Méthode: POST
# Type de Contenu: multipart/form-data
# Champ de fichier: 'file' (nom du champ)
# Valeur du champ: votre fichier image
# ====================================================================