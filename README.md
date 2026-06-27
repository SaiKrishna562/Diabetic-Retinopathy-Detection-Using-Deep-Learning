<div align="center">

# 👁️ Diabetic Retinopathy Detection Using Deep Learning

**A Streamlit web app that classifies diabetic retinopathy severity from retinal fundus images using ensemble deep learning models.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 📖 Overview

**Diabetic Retinopathy (DR)** is a leading cause of preventable blindness caused by damage to the blood vessels in the retina due to prolonged diabetes. Early detection through regular screening of retinal fundus images can prevent severe vision loss — but manual grading by ophthalmologists is time-consuming and requires specialist expertise.

This project is a web-based diagnostic aid that lets a user log in, choose from **four pretrained CNN architectures**, upload a retinal fundus image, and instantly get a **severity classification** for diabetic retinopathy.

> ⚠️ **Disclaimer:** This tool is built for academic/research purposes and is **not** a substitute for professional medical diagnosis. Any classification output should be verified by a qualified ophthalmologist.

## ✨ Features

- 🔐 **User login** — simple authentication before accessing the classifier
- 🧠 **Four CNN model options** — choose between **ResNet**, **InceptionNet**, **DenseNet**, and **AlexNet**, each trained and saved as `.h5` Keras models
- 🖼️ **Image upload & preview** — upload a retinal fundus image directly in the browser
- 📊 **Severity classification** — predicts the DR severity stage from the uploaded image
- ⚡ **Fast inference** — models are pre-trained and loaded directly for real-time prediction
- 🖥️ **Simple Streamlit UI** — no setup needed beyond installing dependencies and running the app

## 🩺 Severity Classes

The models classify retinal images into the standard DR severity grades:

| Class | Stage              |
|-------|---------------------|
| 0     | No DR               |
| 1     | Mild                |
| 2     | Moderate            |
| 3     | Severe              |
| 4     | Proliferative DR    |

## 🛠️ Tech Stack

| Layer            | Technology                                   |
|-------------------|-----------------------------------------------|
| UI                | [Streamlit](https://streamlit.io/)            |
| Deep Learning     | TensorFlow / Keras (`.h5` models)             |
| Models            | ResNet, InceptionNet, DenseNet, AlexNet (CNNs)|
| Image processing  | OpenCV / Pillow                               |
| Language          | Python 3.10+                                  |

## 📂 Project Structure

```
Diabetic-Retinopathy-Detection-Using-Deep-Learning/
├── app.py                  # Streamlit app — login, model selection, image upload, prediction
├── models/
│   ├── resnet_model.h5
│   ├── inception_model.h5
│   ├── densenet_model.h5
│   └── alexnet_model.h5
├── utils/                  # Preprocessing / helper functions (if applicable)
├── requirements.txt        # Python dependencies
└── README.md
```

> Update the structure above to match your actual file/folder names if they differ.

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/SaiKrishna562/Diabetic-Retinopathy-Detection-Using-Deep-Learning.git
cd Diabetic-Retinopathy-Detection-Using-Deep-Learning

# (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### Usage

1. **Log in** with your credentials.
2. **Select a model** — ResNet, InceptionNet, DenseNet, or AlexNet.
3. **Upload a retinal fundus image** (JPG/PNG).
4. Click **Predict** to get the diabetic retinopathy severity classification.

## 🧠 How It Works

1. **Preprocessing** — the uploaded retinal image is resized and normalized to match the input shape expected by the selected CNN.
2. **Model loading** — the chosen `.h5` model (ResNet / InceptionNet / DenseNet / AlexNet) is loaded into memory.
3. **Inference** — the preprocessed image is passed through the CNN, which outputs class probabilities across the 5 DR severity stages.
4. **Result** — the predicted class (and optionally the confidence score) is displayed to the user.

## 📈 Model Training (Summary)

The four CNN architectures were trained separately on a labeled retinal fundus image dataset (e.g. APTOS 2019 / Kaggle DR dataset — update with the dataset you actually used), with the final trained weights exported in Keras `.h5` format for fast loading at inference time in the Streamlit app.

> Add your actual training details here — dataset name & size, train/val/test split, augmentation strategy, and accuracy/F1 per model — once finalized.

## 🗺️ Roadmap / Ideas

- [ ] Add per-model accuracy/confusion matrix comparison in the UI
- [ ] Grad-CAM visualization to highlight regions influencing the prediction
- [ ] Support batch image upload
- [ ] Deploy publicly (Streamlit Community Cloud / HuggingFace Spaces)
- [ ] Replace basic login with proper authentication (e.g. hashed credentials or OAuth)

## 🤝 Contributing

Suggestions and pull requests are welcome — feel free to open an issue first to discuss what you'd like to change.

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 🙋 Author

**Sai Krishna** — B.Tech Final Year, JNTU Hyderabad
