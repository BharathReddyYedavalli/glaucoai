import gradio as gr
from main import predict_image

def create_app():
    demo = gr.Interface(
        fn=predict_image,
        inputs=gr.Image(type="filepath", label="Upload Retinal Image"),
        outputs="text",
        title="GlaucoAI - Glaucoma Detection with Grad-CAM",
        description="Upload a retinal fundus image to detect glaucoma and view Grad-CAM visualization.",
    )
    return demo

app = create_app()

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=8080, show_error=True, inline=False)
