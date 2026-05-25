import os
import gradio as gr
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

IMG_SIZE = 128
model = None


def predict(img):
    global model

    try:
        if model is None:
            model = load_model("deepfake_model.keras")

        img = img.resize((IMG_SIZE, IMG_SIZE))

        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0][0]

        if prediction > 0.5:
            return "REAL IMAGE"
        else:
            return "FAKE IMAGE"

    except Exception as e:
        return f"Error: {str(e)}"


app = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(label="Prediction"),
    title="Deepfake Detector",
    description="Upload an image to check whether it is REAL or FAKE."
)

port = int(os.environ.get("PORT", 10000))

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False
    )