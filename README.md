# 🌟 Glow AI - Skin Type Analyzer

Glow AI is an intelligent skin analysis tool that combines quiz-based insights and deep learning to predict your skin type (dry, normal, or oily) using both answers and an uploaded photo.

---

## 🔍 How It Works
a
1. **Answer a few simple questions** about your skin routine and behavior.
2. **Upload or capture a skin photo** using your camera or device.
3. Glow AI’s trained model (MobileNetV2) and smart logic will:
   - Predict your skin type from the image.
   - Cross-check it with your quiz responses.
   - Provide a final result with personalized skincare tips.

---

## 🛠️ Tech Stack

- `TensorFlow` / `Keras` (MobileNetV2 for image classification)
- `Gradio` for interactive UI
- Python 3.10+

---

## 📦 Files

- `app.py`: Main Gradio app logic
- `glow_ai_mobilenet1.h5`: Trained image classification model (tracked using Git LFS)
- `requirements.txt`: Python dependencies

---

## 📸 Sample

![demo](https://huggingface.co/spaces/HudaHajira/glow-ai/resolve/main/thumbnail.png)

---

## 💡 Future Plans

- Improve quiz logic with more dermatology-based questions
- Add support for more skin conditions (acne, pigmentation)
- Improve model accuracy using a larger dataset

---

Made with 💙 by [Huda Hajira](https://github.com/huda-code)
