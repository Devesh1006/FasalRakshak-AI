#  FasalRakshak AI

## AI-Powered Crop Disease Detection & Evidence-Weighted Outbreak Intelligence

FasalRakshak AI is an AI-powered agricultural web application that helps identify crop diseases from plant/leaf images and provides disease information, crop-health scoring, reliability analysis, monitoring history, and geographically organized outbreak intelligence.

The project is designed around the idea that **an increase in reports does not always mean an increase in independent evidence**.

FasalRakshak therefore combines:

- AI-based crop disease detection
- Prediction confidence
- AI reliability analysis
- Crop Health Score
- Disease symptoms
- Fertilizer guidance
- Prevention recommendations
- Multilingual support
- Crop monitoring history
- Geographic outbreak clustering
- Evidence-weighted outbreak analysis
- Interactive outbreak mapping

---

# 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [Problem Statement](#-problem-statement)
3. [Solution](#-solution)
4. [Features](#-features)
5. [AI Model](#-ai-model)
6. [Supported Classes](#-supported-classes)
7. [Technology Stack](#-technology-stack)
8. [Project Structure](#-project-structure)
9. [System Requirements](#-system-requirements)
10. [Installation](#-installation)
11. [Virtual Environment](#-virtual-environment)
12. [Install Dependencies](#-install-dependencies)
13. [Run the Application](#-run-the-application)
14. [Application Pages](#-application-pages)
15. [How to Use the Application](#-how-to-use-the-application)
16. [Model Testing](#-model-testing)
17. [Model Evaluation](#-model-evaluation)
18. [Dataset Preparation](#-dataset-preparation)
19. [Model Training](#-model-training)
20. [Reference Data](#-reference-data)
21. [Multilingual Support](#-multilingual-support)
22. [Outbreak Mapping](#-outbreak-mapping)
23. [Monitoring History](#-monitoring-history)
24. [Important Files](#-important-files)
25. [Git and GitHub Commands](#-git-and-github-commands)
26. [Troubleshooting](#-troubleshooting)
27. [Future Scope](#-future-scope)
28. [Disclaimer](#-disclaimer)

---

# 🌾 Project Overview

FasalRakshak AI is a Flask-based AI web application for agricultural crop monitoring.

The main workflow is:

```text
Farmer uploads crop/leaf image
              ↓
       Image preprocessing
              ↓
       AI disease detection
              ↓
       Confidence analysis
              ↓
       Reliability analysis
              ↓
       Crop Health Score
              ↓
 Disease symptoms / fertilizer / prevention
              ↓
       Monitoring history
              ↓
    Outbreak intelligence
              ↓
      Geographic map
---
#  Features

## 1.  AI Crop Disease Detection

Upload a crop or leaf image and the trained AI model predicts the most likely disease/class.

---

## 2.  MobileNetV2 Model

The current model uses MobileNetV2 with transfer learning and fine-tuning.

Current configuration:

- Architecture: MobileNetV2
- Input size: 224 × 224
- Classes: 38
- Transfer learning: Yes
- Fine-tuning: Yes
- Data augmentation: Yes

Current validation accuracy:

**97.12%**

Model file:

```text
model/crop_disease_model.keras
```

---

## 3.  Prediction Confidence

The result page displays the model's prediction confidence.

Example:

```text
Tomato Late Blight

Confidence: 91%
```

---

## 4.  AI Reliability Analysis

FasalRakshak does not rely only on the top prediction probability.

The system can consider:

- Prediction confidence
- Prediction margin
- Image quality
- Feature similarity
- Reference examples
- Alternative predictions
- Image/view consistency where available

If the result is considered unreliable, the system can warn the user instead of presenting it as a definitive diagnosis.

---

## 5.  AI Uncertainty Warning

When the system detects an uncertain prediction, it can display an uncertainty warning and recommend taking a clearer image.

---

## 6. Top Predictions

The application can show the top candidate predictions instead of hiding alternative classes.

---

## 7.  Crop Health Score

The application provides a simplified **0–100 Crop Health Score**.

Example:

```text
Crop Health

78 / 100
```

This makes the AI output easier to understand.

---

## 8.  Disease Symptoms

Disease-specific symptoms are shown on the result page.

---

## 9.  Fertilizer Guidance

The system provides general fertilizer/nutritional guidance associated with the detected condition.

---

## 10.  Prevention Guidance

The result page provides practical prevention recommendations.

---

## 11.  Multilingual Interface

Supported languages:

- English
- Hindi
- Marathi

Translation logic is maintained in:

```text
translations.py
```

---

## 12.  Image Upload

The homepage supports image selection and drag-and-drop upload.

Supported formats:

```text
JPG
JPEG
PNG
WEBP
```

Maximum upload size:

```text
5 MB
```

---

## 13.  Interactive Outbreak Map

The project includes a dedicated outbreak intelligence page.

It uses an interactive map to display geographically organized outbreak information.

---

## 14.  Geographic Clustering

Disease reports can be organized into geographic clusters.

The purpose is to identify areas where multiple reports may represent a localized outbreak.

---

## 15.  Evidence-Weighted Outbreak Concept

The system is designed around:

```text
Evidence
   +
Geographic Diversity
   +
Disease Confidence
   +
Time Trend
   ↓
Outbreak Signal
```

rather than:

```text
Raw Report Count
       ↓
Outbreak
```

---

## 16.  Genuine Outbreak Demonstration

The outbreak page includes a demonstration scenario representing geographically distributed evidence.

---

## 17.  Panic Reporting Demonstration

The outbreak page also contains a demonstration of concentrated/panic reporting.

This helps explain the CX0603 concept to judges.

---

## 18.  Outbreak Trends

The outbreak dashboard provides trend information to visualize report activity over time.

---

## 19.  Monitoring History

Previous crop scans can be stored in the local SQLite database.

History includes information such as:

- Date/time
- Crop
- Disease
- Confidence
- Crop Health Score

---

## 20.  History Statistics

The history page provides summary information such as:

- Total scans
- Average confidence
- Healthy scans
- Disease scans

---

## 21.  Monitoring Chart

Historical crop-health information can be visualized using the history chart.

---

#  AI Model

## Architecture

```text
MobileNetV2
     ↓
Feature Extraction
     ↓
Classification Head
     ↓
38 Crop/Disease Classes
```

The model uses transfer learning and fine-tuning.

---

## Training Stages

### Stage 1

The pretrained feature extractor is used with a classification head.

### Stage 2

Selected upper layers are fine-tuned using a lower learning rate.

---

## Model Performance

Current validation result:

```text
Validation Accuracy: 97.12%
Validation Loss:     0.0798
```

The validation result comes from the PlantVillage-based dataset used during development.

Real-world performance can differ because field photographs can have:

- Different lighting
- Complex backgrounds
- Different camera quality
- Different crop varieties
- Multiple symptoms
- Unseen disease appearances

---


---

#  Technology Stack

## Backend

- Python
- Flask

## AI / Machine Learning

- TensorFlow
- Keras
- MobileNetV2
- NumPy
- Pillow

## Frontend

- HTML
- CSS
- JavaScript

## Maps

- Leaflet

## Database

- SQLite

## Dataset

- PlantVillage

---

#  Project Structure

```text
FasalRakshak-AI/
│
├── app.py
├── disease_data.py
├── translations.py
│
├── train_model.py
├── prepare_dataset.py
├── evaluate_model.py
├── test_model.py
├── build_reference.py
├── check_dataset.py
│
├── requirements.txt
├── README.md
├── .gitignore
│
├── model/
│   ├── classes.json
│   ├── crop_disease_model.keras
│   └── reference_data.npz
│
├── static/
│   ├── fasalrakshak-logo.jpeg
│   ├── fasalrakshak-ai-visual.png
│   ├── style.css
│   └── uploads/
│
└── templates/
    ├── index.html
    ├── result.html
    ├── history.html
    └── outbreak_map.html
```

---



#  Installation

## 1. Clone the Repository

Open PowerShell:

```powershell
git clone https://github.com/Devesh1006/FasalRakshak-AI.git
```

Enter the project:

```powershell
cd FasalRakshak-AI
```

---

#  Virtual Environment

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

After activation, the terminal should show something similar to:

```text
(.venv) PS C:\...\FasalRakshak-AI>
```

---

## If PowerShell Blocks Activation

If Windows shows an execution-policy error, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

#  Install Dependencies

Make sure the virtual environment is active.

Then:

```powershell
pip install -r requirements.txt
```

If `requirements.txt` is unavailable or you need the main runtime packages manually:

```powershell
pip install flask tensorflow numpy pillow opencv-python
```

Verify important packages:

```powershell
pip show flask
pip show tensorflow
pip show numpy
pip show pillow
```

---

#  Run the Application

Make sure you are inside the project directory:

```powershell
cd FasalRakshak-AI
```

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Run:

```powershell
python app.py
```

You should see Flask start on:

```text
http://127.0.0.1:5000
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

---

#  Application Pages

## Home

```text
http://127.0.0.1:5000/
```

Use this page to:

- Upload a crop image
- Select a language
- Start disease analysis

---

## Result

The result page is generated after uploading an image.

It displays:

- Crop
- Disease
- Confidence
- Reliability
- Crop Health Score
- Top predictions
- Symptoms
- Fertilizer guidance
- Prevention guidance

---

## Monitoring History

```text
http://127.0.0.1:5000/history
```

Use this page to review previous scans.

---

## Outbreak Map

```text
http://127.0.0.1:5000/outbreak-map
```

Use this page to explore:

- Outbreak clusters
- Geographic reports
- Trend information
- Evidence demonstrations
- Interactive map

---

#  How to Use the Application

## Step 1

Open:

```text
http://127.0.0.1:5000
```

## Step 2

Choose:

```text
Upload Crop Image
```

## Step 3

Select a JPG, PNG, JPEG, or WEBP image.

Maximum size:

```text
5 MB
```

## Step 4

The application automatically analyzes the image.

## Step 5

Review:

```text
Disease
Confidence
Reliability
Crop Health Score
Symptoms
Fertilizer guidance
Prevention
```

## Step 6

For outbreak information, open:

```text
http://127.0.0.1:5000/outbreak-map
```

---

#  Model Testing

The project contains:

```text
test_model.py
```

To run the model test:

```powershell
python test_model.py
```

This can be used to verify that the trained model and class mapping are functioning.

---

#  Model Evaluation

The project contains:

```text
evaluate_model.py
```

Run:

```powershell
python evaluate_model.py
```

This is used for model evaluation.

---

#  Dataset Preparation

Dataset preparation is handled by:

```text
prepare_dataset.py
```

Before preparing or retraining the model, the required dataset must be available locally.

The dataset is intentionally excluded from GitHub because of its size.

The `.gitignore` contains:

```text
dataset/
```

---

#  Check Dataset

Use:

```powershell
python check_dataset.py
```

This script helps verify whether the dataset structure is detected correctly.

---

#  Build Reference Data

The reliability layer uses reference feature data.

Reference generation is handled by:

```text
build_reference.py
```

Run:

```powershell
python build_reference.py
```

The generated reference file is:

```text
model/reference_data.npz
```

---

#  Train the Model

Model training is handled by:

```text
train_model.py
```

Run:

```powershell
python train_model.py
```

Training may take significant time depending on:

- CPU/GPU
- RAM
- Dataset size
- TensorFlow configuration

The trained model is saved as:

```text
model/crop_disease_model.keras
```

Class information is saved as:

```text
model/classes.json
```

---

#  Important Training Note

The existing trained model is already included in the repository.

You normally do **not** need to train the model just to run the web application.

Training is only required when you want to:

- Change the dataset
- Add new classes
- Improve the model
- Experiment with the architecture
- Retrain using new field data

---

#  Multilingual Support

The project currently supports:

```text
English
Hindi
Marathi
```

Translation configuration is stored in:

```text
translations.py
```

The language system covers interface text and disease-related information where translations are available.

---

#  Outbreak Mapping

The outbreak mapping functionality is available at:

```text
http://127.0.0.1:5000/outbreak-map
```

The system demonstrates the concept of evidence-weighted geographic outbreak intelligence.

The key concept is:

```text
Independent Evidence
        +
Geographic Distribution
        +
AI Confidence
        ↓
Outbreak Signal
```

The project also includes demonstration scenarios for:

```text
Genuine Outbreak
Panic Reporting
```

These are demonstration features for the hackathon concept and should not be interpreted as real epidemiological outbreak detection.

---

#  Monitoring History

History is stored locally using SQLite.

The local database file is:

```text
cropcare.db
```

The database is intentionally excluded from GitHub.

The history page can display:

- Total scans
- Average confidence
- Healthy scans
- Disease scans
- Individual scan records
- Health scores
- Historical chart

---


---

#  Git and GitHub Commands

## Check Git Version

```powershell
git --version
```

---

## Check Repository Status

```powershell
git status
```

---

## Check Current Branch

```powershell
git branch
```

---

## Check Remote Repository

```powershell
git remote -v
```

---

## Pull Latest Changes

```powershell
git pull origin main
```

---

## Add Changes

```powershell
git add .
```

---

## Check Staged Changes

```powershell
git status
```

---

## Commit Changes

```powershell
git commit -m "Describe your changes"
```

Example:

```powershell
git commit -m "Improve outbreak map UI"
```

---

## Push Changes

```powershell
git push origin main
```

---

## Complete Update Workflow

Whenever changes are made:

```powershell
git status
git add .
git commit -m "Update FasalRakshak AI"
git push origin main
```

---

#  Updating the Project on Another Computer

Clone the repository:

```powershell
git clone https://github.com/Devesh1006/FasalRakshak-AI.git
```

Enter it:

```powershell
cd FasalRakshak-AI
```

Create environment:

```powershell
python -m venv .venv
```

Activate:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install:

```powershell
pip install -r requirements.txt
```

Run:

```powershell
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

---




---

#  Environment Variables / Secrets

Never commit:

```text
.env
API keys
Passwords
Private credentials
Access tokens
```

If future services such as WhatsApp APIs, weather APIs, or cloud databases are added, their credentials should be stored using environment variables.

---

#  Troubleshooting

## Problem: `git` is not recognized

Install Git for Windows and restart VS Code.

Check:

```powershell
git --version
```

---

## Problem: `python` is not recognized

Check:

```powershell
python --version
```

If Python is installed but not recognized, add Python to PATH or reinstall Python with:

```text
Add Python to PATH
```

enabled.

---

## Problem: Flask is missing

Error:

```text
ModuleNotFoundError: No module named 'flask'
```

Run:

```powershell
pip install flask
```

Or install everything:

```powershell
pip install -r requirements.txt
```

---

## Problem: PIL is missing

Error:

```text
ModuleNotFoundError: No module named 'PIL'
```

Run:

```powershell
pip install Pillow
```

---

## Problem: NumPy is missing

Error:

```text
ModuleNotFoundError: No module named 'numpy'
```

Run:

```powershell
pip install numpy
```

---

## Problem: OpenCV is missing

Run:

```powershell
pip install opencv-python
```

---

## Problem: TensorFlow is missing

Run:

```powershell
pip install tensorflow
```

---

## Problem: Wrong Python Environment

Check:

```powershell
where python
```

The active environment should point to something similar to:

```text
...\FasalRakshak-AI\.venv\Scripts\python.exe
```

Activate the environment again:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## Problem: Port 5000 is Already in Use

Stop the existing Flask process or change the port in `app.py`.

The normal application URL is:

```text
http://127.0.0.1:5000
```

---

## Problem: Model File Not Found

Check:

```powershell
Get-ChildItem .\model\
```

You should have:

```text
classes.json
crop_disease_model.keras
reference_data.npz
```

---

## Problem: Dataset Not Found

The dataset is not included in the GitHub repository.

This is intentional.

The project can run using the already trained model without downloading the training dataset again.

The dataset is only required for model training/evaluation workflows.

---

#  Future Scope

Planned future improvements include:

##  WhatsApp Alerts

Send outbreak warnings to registered farmers.

##  Hyperlocal Geo-Fencing

Notify farmers within a configurable distance from a verified outbreak cluster.

##  Weather Integration

Combine weather conditions with disease reports.

##  Advanced Outbreak Risk Score

Generate an outbreak risk score using:

- Evidence
- Geographic diversity
- AI confidence
- Weather
- Time trends

##  Community Verification

Allow nearby farmers to confirm or dispute an outbreak signal.

## Voice Assistant

Support farmer queries through Hindi and Marathi voice interaction.

##  Offline Mode

Store reports locally and synchronize them when connectivity returns.

##  Satellite Integration

Use satellite-derived crop/vegetation information for regional risk monitoring.

## Mobile Application

Extend the platform to Android/mobile devices.



#  FasalRakshak AI

### Turning crop images and farmer observations into agricultural intelligence.

**AI Diagnosis → Reliability → Crop Health → Evidence → Geographic Intelligence → Early Warning**
#   F a s a l r a k s h a k - A I  
 