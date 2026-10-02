🌾 FasalRakshak AI

AI-Powered Crop Disease Detection & Evidence-Weighted Outbreak Intelligence

FasalRakshak AI is an AI-powered agricultural web application that helps identify crop diseases from plant/leaf images and provides disease information, crop-health scoring, reliability analysis, monitoring history, and geographically organized outbreak intelligence.

Core idea: An increase in reports does not always mean an increase in independent evidence.

The platform combines AI-based disease detection with confidence analysis, crop-health assessment, monitoring history, and evidence-weighted geographic outbreak intelligence.

📌 Table of Contents

Project Overview

Problem Statement

Solution

Key Features

AI Model

Supported Classes

Technology Stack

Project Structure

System Requirements

Installation

Run the Application

How to Use

Application Pages

Model Testing & Evaluation

Dataset & Training

Multilingual Support

Outbreak Intelligence

Monitoring History

Git & GitHub Workflow

Troubleshooting

Future Scope

Disclaimer

🌱 Project Overview

FasalRakshak AI follows this workflow:

Farmer uploads crop/leaf image
          ↓
Image preprocessing
          ↓
AI disease detection
          ↓
Confidence & reliability analysis
          ↓
Crop Health Score
          ↓
Symptoms / Fertilizer / Prevention guidance
          ↓
Monitoring history
          ↓
Evidence-weighted outbreak intelligence
          ↓
Interactive geographic map

🎯 Problem Statement

Crop diseases can spread before farmers are able to recognize a wider pattern.

A single disease report may indicate an isolated case, while many reports from the same area may be caused by repeated reporting of the same local observation. Therefore, simply counting reports may not provide enough evidence of a developing outbreak.

FasalRakshak AI explores an evidence-weighted approach that considers:

Disease prediction confidence

Geographic distribution

Independent evidence

Time trends

Image/reliability signals

💡 Solution

FasalRakshak AI combines crop disease detection with monitoring and outbreak intelligence.

Instead of treating:

More Reports → Outbreak

as sufficient evidence, the project uses the concept:

Evidence
+ Geographic Diversity
+ Disease Confidence
+ Time Trend
        ↓
Outbreak Signal

The outbreak functionality is designed as a demonstration of this concept rather than a real epidemiological diagnosis system.

🚀 Key Features

1. AI Crop Disease Detection

Upload a crop or leaf image and the trained AI model predicts the most likely disease/class.

2. Prediction Confidence

The result page displays the model's prediction confidence.

Example:

Tomato Late Blight
Confidence: 91%

3. AI Reliability Analysis

The application can consider multiple signals instead of relying only on the top prediction probability:

Prediction confidence

Prediction margin

Image quality

Feature similarity

Reference examples

Alternative predictions

Image/view consistency where available

When a result is considered unreliable, the system can warn the user instead of presenting it as a definitive diagnosis.

4. Uncertainty Warning

Uncertain predictions can trigger a warning recommending a clearer image.

5. Top Predictions

The application can display alternative candidate predictions rather than hiding them.

6. Crop Health Score

A simplified 0–100 Crop Health Score helps make the AI output easier to understand.

Example:

Crop Health: 78 / 100

7. Disease Symptoms

Disease-specific symptoms are displayed on the result page.

8. Fertilizer Guidance

The application provides general fertilizer/nutritional guidance associated with the detected condition.

9. Prevention Guidance

The result page provides practical prevention recommendations.

10. Multilingual Interface

Supported languages:

English

Hindi

Marathi

Translation logic is maintained in translations.py.

11. Image Upload

Supported image formats:

JPG
JPEG
PNG
WEBP

Maximum upload size:

5 MB

12. Interactive Outbreak Map

The outbreak intelligence page provides an interactive geographic map showing organized outbreak information and demonstration scenarios.

13. Geographic Clustering

Disease reports can be organized into geographic clusters to explore areas where multiple reports may represent a localized outbreak.

14. Monitoring History

Previous crop scans can be stored locally using SQLite.

History can include:

Date/time

Crop

Disease

Confidence

Crop Health Score

15. History Statistics

The history page can show:

Total scans

Average confidence

Healthy scans

Disease scans

Historical crop-health information

🤖 AI Model

Architecture

The current model uses MobileNetV2 with transfer learning and fine-tuning.

MobileNetV2
     ↓
Feature Extraction
     ↓
Classification Head
     ↓
38 Crop/Disease Classes

Current Configuration

Property

Value

Architecture

MobileNetV2

Input Size

224 × 224

Number of Classes

38

Transfer Learning

Yes

Fine-Tuning

Yes

Data Augmentation

Yes

Validation Accuracy

97.12%

Validation Loss

0.0798

Model file:

model/crop_disease_model.keras

The reported validation result comes from the PlantVillage-based dataset used during development. Real-world field performance can differ because field photographs may have different lighting, backgrounds, camera quality, crop varieties, multiple symptoms, and unseen disease appearances.

🌿 Supported Classes

The model currently supports 38 crop/disease classes.

Class mapping is stored in:

model/classes.json

🛠️ Technology Stack

Backend

Python

Flask

AI / Machine Learning

TensorFlow

Keras

MobileNetV2

NumPy

Pillow

Frontend

HTML

CSS

JavaScript

Maps

Leaflet

Database

SQLite

Dataset

PlantVillage

📁 Project Structure

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

💻 System Requirements

Recommended environment:

Windows / Linux / macOS

Python 3.x

Git

Internet connection for initial dependency installation

Sufficient RAM and storage for TensorFlow and the project model

The existing trained model is included in the repository, so the dataset is not required just to run the web application.

⚙️ Installation

1. Clone the Repository

git clone https://github.com/Devesh1006/FasalRakshak-AI.git

Enter the project directory:

cd FasalRakshak-AI

2. Create a Virtual Environment

python -m venv .venv

Activate it:

.\.venv\Scripts\Activate.ps1

You should see something similar to:

(.venv) PS C:\...\FasalRakshak-AI>

If PowerShell Blocks Activation

Run:

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

Then activate again:

.\.venv\Scripts\Activate.ps1

3. Install Dependencies

pip install -r requirements.txt

If required, the main runtime packages can also be installed manually:

pip install flask tensorflow numpy pillow opencv-python

▶️ Run the Application

Make sure the virtual environment is active:

.\.venv\Scripts\Activate.ps1

Then run:

python app.py

The application should start at:

http://127.0.0.1:5000

Open that address in your browser.

🖥️ Application Pages

Home

http://127.0.0.1:5000/

Use this page to:

Upload a crop image

Select a language

Start disease analysis

Result

The result page displays information such as:

Crop

Disease

Confidence

Reliability

Crop Health Score

Top predictions

Symptoms

Fertilizer guidance

Prevention guidance

Monitoring History

http://127.0.0.1:5000/history

Review previous crop scans and history statistics.

Outbreak Map

http://127.0.0.1:5000/outbreak-map

Explore:

Geographic reports

Outbreak clusters

Trend information

Evidence demonstrations

Interactive map

📸 How to Use

Open the application.

Choose Upload Crop Image.

Select a JPG, JPEG, PNG, or WEBP image.

Keep the image size within the 5 MB upload limit.

The application analyzes the image.

Review the disease prediction and confidence.

Review reliability, Crop Health Score, symptoms, fertilizer guidance, and prevention recommendations.

Open the history page to review previous scans.

Open the outbreak map to explore the geographic intelligence demonstration.

🧪 Model Testing & Evaluation

Test the Model

The project contains:

test_model.py

Run:

python test_model.py

This can be used to verify that the trained model and class mapping are functioning.

Evaluate the Model

The project contains:

evaluate_model.py

Run:

python evaluate_model.py

This is used for model evaluation.

🗂️ Dataset & Training

Dataset Preparation

Dataset preparation is handled by:

prepare_dataset.py

Before preparing or retraining the model, the required dataset must be available locally.

The dataset is intentionally excluded from GitHub because of its size.

The .gitignore contains:

dataset/

Check Dataset

python check_dataset.py

Build Reference Data

Reference feature generation is handled by:

build_reference.py

Run:

python build_reference.py

Generated file:

model/reference_data.npz

Train the Model

Training is handled by:

train_model.py

Run:

python train_model.py

The trained model is saved as:

model/crop_disease_model.keras

Class information is saved as:

model/classes.json

Training is not normally required just to run the web application. Retraining is useful when changing the dataset, adding classes, improving the model, experimenting with the architecture, or training with new field data.

🌐 Multilingual Support

The current interface supports:

🇬🇧 English

🇮🇳 Hindi

🇮🇳 Marathi

Translation configuration is maintained in:

translations.py

🗺️ Outbreak Intelligence

The outbreak intelligence functionality demonstrates an evidence-weighted approach to geographic disease reporting.

Core Concept

Independent Evidence
        +
Geographic Distribution
        +
AI Confidence
        ↓
Outbreak Signal

The project also includes demonstration scenarios for:

Genuine Outbreak

Panic Reporting

These scenarios are intended to demonstrate the project concept and should not be interpreted as real epidemiological outbreak detection.

Important Distinction

Raw Report Count
       ≠
Independent Evidence

The system is designed to explore how geographic diversity, confidence, and time trends can provide additional context.

📊 Monitoring History

History is stored locally using SQLite.

Database file:

cropcare.db

The database is intentionally excluded from GitHub.

The history page can display:

Total scans

Average confidence

Healthy scans

Disease scans

Individual scan records

Crop Health Scores

Historical chart

🔄 Git & GitHub Workflow

Check Repository Status

git status

Pull Latest Changes

git pull origin main

Add Changes

git add .

Commit Changes

git commit -m "Update FasalRakshak AI"

Push Changes

git push origin main

Complete Workflow

Whenever you modify the project:

git status
git add .
git commit -m "Update FasalRakshak AI"
git push origin main

💻 Running the Project on Another Computer

Clone the repository:

git clone https://github.com/Devesh1006/FasalRakshak-AI.git

Enter the project:

cd FasalRakshak-AI

Create the environment:

python -m venv .venv

Activate it:

.\.venv\Scripts\Activate.ps1

Install dependencies:

pip install -r requirements.txt

Run the application:

python app.py

Then open:

http://127.0.0.1:5000

🔐 Environment Variables & Secrets

Never commit sensitive information such as:

.env
API keys
Passwords
Private credentials
Access tokens

If future services such as WhatsApp APIs, weather APIs, or cloud databases are added, store their credentials using environment variables.

🛠️ Troubleshooting

git is not recognized

Install Git for Windows and restart your terminal.

Check:

git --version

python is not recognized

Check:

python --version

If Python is installed but not recognized, add Python to PATH or reinstall Python with Add Python to PATH enabled.

Flask is missing

ModuleNotFoundError: No module named 'flask'

Run:

pip install flask

Or:

pip install -r requirements.txt

PIL is missing

ModuleNotFoundError: No module named 'PIL'

Run:

pip install Pillow

NumPy is missing

pip install numpy

OpenCV is missing

pip install opencv-python

TensorFlow is missing

pip install tensorflow

Wrong Python Environment

Check:

where python

The active environment should point to something similar to:

...\FasalRakshak-AI\.venv\Scripts\python.exe

Activate again if necessary:

.\.venv\Scripts\Activate.ps1

Port 5000 is Already in Use

Stop the existing Flask process or change the port in app.py.

Normal application URL:

http://127.0.0.1:5000

Model File Not Found

Check:

Get-ChildItem .\model\

The model directory should contain:

classes.json
crop_disease_model.keras
reference_data.npz

Dataset Not Found

The dataset is not included in GitHub intentionally.

The application can run using the existing trained model without downloading the training dataset again.

The dataset is required only for training/evaluation workflows.

🔮 Future Scope

Planned improvements include:

📱 WhatsApp Alerts

Send outbreak warnings to registered farmers.

📍 Hyperlocal Geo-Fencing

Notify farmers within a configurable distance from a verified outbreak cluster.

🌦️ Weather Integration

Combine weather conditions with disease reports.

📈 Advanced Outbreak Risk Score

Combine:

Evidence

Geographic diversity

AI confidence

Weather

Time trends

👨‍🌾 Community Verification

Allow nearby farmers to confirm or dispute an outbreak signal.

🎙️ Voice Assistant

Support farmer queries through Hindi and Marathi voice interaction.

📴 Offline Mode

Store reports locally and synchronize them when connectivity returns.

🛰️ Satellite Integration

Use satellite-derived crop/vegetation information for regional risk monitoring.

📲 Mobile Application

Extend the platform to Android/mobile devices.

⚠️ Disclaimer

FasalRakshak AI is a project prototype for crop disease detection, monitoring, and evidence-weighted outbreak intelligence.

AI predictions and outbreak demonstrations should not be treated as a substitute for professional agricultural or plant-disease diagnosis.

Real-world model performance may differ from validation performance depending on image quality, environment, crop variety, disease appearance, and other field conditions.

🌾 FasalRakshak AI

Turning crop images and farmer observations into agricultural intelligence.

AI Diagnosis
     ↓
Reliability
     ↓
Crop Health
     ↓
Evidence
     ↓
Geographic Intelligence
     ↓
Early Warning

👨‍💻 Project Repository

GitHub: https://github.com/Devesh1006/FasalRakshak-AI