import os
import gradio as gr

def predict(img):
    return "TEST OK"

app = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Test"
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.launch(
        server_name="0.0.0.0",
        server_port=port
    )