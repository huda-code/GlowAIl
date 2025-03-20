import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load model
model = tf.keras.models.load_model("glow_ai_mobilenet1.h5")
class_labels = ['dry', 'normal', 'oily']

# Define quiz analysis logic
def analyze_quiz(q1, q2, q3, q4, q5, q6, q7):
    if q1 == 'a' or q2 == 'yes' or q5 == 'yes':
        return 'dry'
    elif q1 in ['b', 'c'] or q6 == 'yes':
        return 'oily'
    elif q3 == 'yes' or q4 == 'yes' or q7 == 'yes':
        return 'sensitive'
    else:
        return 'normal'

# Define image + quiz analysis
def predict_skin_type(img, q1, q2, q3, q4, q5, q6, q7):
    quiz_result = analyze_quiz(q1, q2, q3, q4, q5, q6, q7)
    
    image_result = "None"
    if img is not None:
        img = img.resize((224, 224))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
        pred = model.predict(img_array)
        image_result = class_labels[np.argmax(pred)]

    suggestion_map = {
        "dry": "Use deep moisturizers and hydrate often.",
        "oily": "Use oil-free products, cleanse gently.",
        "normal": "Keep your skincare simple.",
        "sensitive": "Use fragrance-free, patch test products."
    }

    result_text = f"📌 Quiz: {quiz_result} | 🖼️ Image: {image_result}"
    suggestion = suggestion_map.get(quiz_result, "Use balanced care.")
    return result_text, f"💡 Suggestion: {suggestion}"
with gr.Blocks() as demo:
    gr.Markdown("## 🌟 Glow AI - Skin Type Analyzer")
    gr.Markdown("Answer a few questions and upload a photo to get your skin type!")

    img_input = gr.Image(type="pil", label="Upload or Take Skin Photo")

    with gr.Group():
        q1 = gr.Radio(["a", "b", "c", "d"], label="1. How does your skin feel during the day?\na=Tight, b=Oily, c=Shiny, d=Reacts", interactive=True)
        q2 = gr.Radio(["yes", "no"], label="2. Does your skin feel tight after washing?")
        q3 = gr.Radio(["yes", "no"], label="3. Does your skin react to products?")
        q4 = gr.Radio(["yes", "no"], label="4. Do you get flare-ups (weather, food, etc)?")
        q5 = gr.Radio(["yes", "no"], label="5. Does your skin feel flaky or rough?")
        q6 = gr.Radio(["yes", "no"], label="6. Does your skin feel greasy after a few hours?")
        q7 = gr.Radio(["yes", "no"], label="7. Does your skin often appear red or irritated?")

    btn = gr.Button("Submit")
    result_text = gr.Textbox(label="Output", interactive=False)
    suggestion = gr.Textbox(label="Skincare Suggestion", interactive=False)

    btn.click(fn=predict_skin_type, inputs=[img_input, q1, q2, q3, q4, q5, q6, q7], outputs=[result_text, suggestion])

demo.launch()
