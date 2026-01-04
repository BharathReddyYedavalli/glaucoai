import gradio as gr
from main import predict_image

demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="filepath", label="Upload Retinal Image"),
    outputs="text",
    title="GlaucoAI - Glaucoma Detection with Grad-CAM",
    description="Upload a retinal image to detect glaucoma and view Grad-CAM visualization."
)

demo.launch()
