import os
import gradio as gr
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

print("Starting app...")

IMG_SIZE = 128

print("Loading model...")
model = load_model("deepfake_model.keras")
print("Model loaded successfully")


def predict(img):
    try:
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


print("Creating Gradio interface...")

app = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(label="Prediction"),
    title="Deepfake Detector",
    description="Upload an image to check whether it is REAL or FAKE."
)

print("Launching app...")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))

    app.launch(
        server_name="0.0.0.0",
        server_port=port
    )