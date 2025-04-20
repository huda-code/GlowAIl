import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import random

# Load model
model = tf.keras.models.load_model("glow_ai_mobilenet1.h5")
class_labels = ['dry', 'normal', 'oily']

# Improved quiz logic with scoring system including "sometimes"
def analyze_quiz(q1, q2, q3, q4, q5, q6, q7):
    scores = {"dry": 0, "oily": 0, "sensitive": 0, "normal": 0}

    # Dry indicators
    if q1 == 'a':
        scores["dry"] += 1
    if q2 == 'yes':
        scores["dry"] += 1
    elif q2 == 'sometimes':
        scores["dry"] += 0.5
    if q5 == 'yes':
        scores["dry"] += 1
    elif q5 == 'sometimes':
        scores["dry"] += 0.5

    # Oily indicators
    if q1 in ['b', 'c']:
        scores["oily"] += 1
    if q6 == 'yes':
        scores["oily"] += 1
    elif q6 == 'sometimes':
        scores["oily"] += 0.5

    # Sensitive indicators
    if q3 == 'yes':
        scores["sensitive"] += 1
    elif q3 == 'sometimes':
        scores["sensitive"] += 0.5
    if q4 == 'yes':
        scores["sensitive"] += 1
    elif q4 == 'sometimes':
        scores["sensitive"] += 0.5
    if q7 == 'yes':
        scores["sensitive"] += 1
    elif q7 == 'sometimes':
        scores["sensitive"] += 0.5

    # Normal indicators
    if q1 == 'd':
        scores["normal"] += 1
    if all(ans == 'no' for ans in [q2, q3, q4, q5, q6, q7]):
        scores["normal"] += 2  # strong indicator for normal

    return max(scores, key=scores.get)

# Final prediction combining quiz and image results
def predict_skin_type(img, q1, q2, q3, q4, q5, q6, q7):
    quiz_result = analyze_quiz(q1, q2, q3, q4, q5, q6, q7)
    image_result = None

    if img is not None:
        img = img.resize((224, 224))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
        pred = model.predict(img_array)
        image_result = class_labels[np.argmax(pred)]

    # Combine logic: prioritize agreement or lean toward quiz
    if image_result == quiz_result:
        final_result = quiz_result
    elif image_result == "normal":
        final_result = quiz_result
    elif quiz_result == "normal":
        final_result = image_result
    else:
        final_result = quiz_result

    # Suggestion map with randomized options
    suggestion_map = {
        "dry": [
            "Use raw honey as a hydrating mask for 15 mins, increase omega-3 intake through walnuts or flaxseeds, and avoid long hot showers.",
            "Apply coconut oil overnight for deep moisture, eat more avocados and drink plenty of water, and use a humidifier if your room air is dry.",
            "Try a mashed banana and milk mask weekly, include hydrating fruits like cucumbers and oranges in your meals, and choose a cream-based cleanser over foam ones."
        ],
        "oily": [
            "Apply a clay mask (like multani mitti) twice a week to absorb excess oil, reduce fried and processed food in your diet, and always use a gel-based, non-comedogenic moisturizer.",
            "Dab diluted tea tree oil on oily areas to reduce acne, include zinc-rich foods like pumpkin seeds and chickpeas, and wash your face twice a day only — not more.",
            "Use aloe vera gel as a light moisturizer to calm oil production, eat more green leafy vegetables and avoid sugary drinks, and change pillowcases often to avoid bacteria buildup."
        ],
        "normal": [
            "Use rose water as a natural toner daily, maintain a balanced diet with proteins, fruits, and fiber, and still moisturize twice a day for long-term skin health.",
            "Apply yogurt and honey mask once a week for glow, eat fermented foods like yogurt or kimchi for gut-skin health, and wear SPF daily even indoors.",
            "Use papaya pulp as a mild exfoliant weekly, drink 8+ glasses of water daily, and avoid touching your face often to keep it blemish-free."
        ],
        "sensitive": [
            "Apply chilled cucumber slices to soothe redness, stick to a bland diet avoiding too many spices, and patch-test any new skincare product.",
            "Use colloidal oatmeal paste to calm irritation, eat antioxidant-rich berries to fight inflammation, and always go for fragrance-free skincare.",
            "Dab chamomile tea (cooled) with cotton on sensitive areas, include turmeric in meals for its anti-inflammatory effects, and avoid over-exfoliating or using harsh scrubs."
        ]
    }

    result_text = f"✅ Skin Type: {final_result.capitalize()}"
    suggestion = random.choice(suggestion_map.get(final_result, ["Use balanced care."]))
    return result_text, f"💡 Suggestion: {suggestion}"

# Gradio UI
with gr.Blocks() as demo:
    with gr.Column(visible=True) as consent_box:
        gr.Image(value="logo.png", show_label=False)
        gr.Markdown("## Privacy Consent")
        gr.Markdown(
            "Our Skin Analyzer is AI-powered and your privacy matters to us. "
            "**Our Skin Analyzer is AI-powered and your privacy matters to us. Your images and data are never saved, stored, or shared** and are only used temporarily to analyze your skin.Your images and data are securely processed and will never be misused or shared with third parties."
        )
        agree = gr.Button("I Agree")
        disagree = gr.Button("I Disagree")

    with gr.Column(visible=False) as analyzer_ui:
        gr.Markdown("## 🌟 Glow AI - Skin Type Analyzer")
        gr.Markdown("Answer a few questions and upload a photo to get your skin type!")

        img_input = gr.Image(type="pil", label="Upload or Take Skin Photo")

        with gr.Group():
            q1 = gr.Radio(["a", "b", "c", "d"], label="1. How does your skin feel during the day?\na=Tight, b=Oily, c=Shiny, d=Reacts", interactive=True)
            q2 = gr.Radio(["yes", "no", "sometimes"], label="2. Does your skin feel tight after washing?")
            q3 = gr.Radio(["yes", "no", "sometimes"], label="3. Does your skin react to products?")
            q4 = gr.Radio(["yes", "no", "sometimes"], label="4. Do you get flare-ups (weather, food, etc)?")
            q5 = gr.Radio(["yes", "no", "sometimes"], label="5. Does your skin feel flaky or rough?")
            q6 = gr.Radio(["yes", "no", "sometimes"], label="6. Does your skin feel greasy after a few hours?")
            q7 = gr.Radio(["yes", "no", "sometimes"], label="7. Does your skin often appear red or irritated?")

        btn = gr.Button("Submit")
        result_text = gr.Textbox(label="Output", interactive=False)
        suggestion = gr.Textbox(label="Skincare Suggestion", interactive=False)

        btn.click(fn=predict_skin_type,
                  inputs=[img_input, q1, q2, q3, q4, q5, q6, q7],
                  outputs=[result_text, suggestion])

    # Consent logic
    agree.click(lambda: (gr.update(visible=False), gr.update(visible=True)),
                outputs=[consent_box, analyzer_ui])
    disagree.click(lambda: exit(), inputs=[], outputs=[])

demo.launch()
