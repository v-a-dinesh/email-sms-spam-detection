# 📧 Email/SMS Spam Classifier

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://email-sms-spam-detection-vad.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![NLTK](https://img.shields.io/badge/NLTK-3.9.1-green.svg)](https://www.nltk.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange.svg)](https://scikit-learn.org/)

A machine learning-powered web application that classifies emails and SMS messages as spam or legitimate using Natural Language Processing (NLP) techniques.

🔗 **Live Demo**: [https://email-sms-spam-detection-vad.streamlit.app/](https://email-sms-spam-detection-vad.streamlit.app/)

## 📋 Table of Contents

- [Features](#-features)
- [Demo](#-demo)
- [Technologies Used](#-technologies-used)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Model Details](#-model-details)
- [Deployment](#-deployment)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

## ✨ Features

- **Real-time Classification**: Instantly classify messages as spam or not spam
- **User-friendly Interface**: Clean and intuitive Streamlit-based web interface
- **High Accuracy**: Trained on extensive dataset for reliable predictions
- **Text Preprocessing**: Advanced NLP preprocessing including:
  - Lowercase conversion
  - Tokenization
  - Stopword removal
  - Stemming
- **Fast Predictions**: Optimized model for quick response times

## 🎥 Demo

![Spam Classifier Demo](https://via.placeholder.com/800x400?text=Spam+Classifier+Demo)

### How to Use:

1. Visit the [live application](https://email-sms-spam-detection-vad.streamlit.app/)
2. Enter your email/SMS text in the text area
3. Click the "Predict" button
4. View the classification result

### Example Messages:

**Spam Message**:
```
Congratulations! You've won a $1000 gift card. Click here to claim now: bit.ly/claim-prize
```

**Legitimate Message**:
```
Hi John, just wanted to remind you about our meeting tomorrow at 3 PM. See you then!
```

## 🛠 Technologies Used

- **Frontend**: Streamlit
- **Backend**: Python 3.9+
- **Machine Learning**: 
  - scikit-learn (SVM Classifier)
  - TF-IDF Vectorization
- **NLP**: NLTK (Natural Language Toolkit)
- **Deployment**: Streamlit Community Cloud

## 💻 Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Local Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/v-a-dinesh/email-sms-spam-detection.git
   cd email-sms-spam-detection
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data**:
   ```python
   python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')"
   ```

5. **Run the application**:
   ```bash
   streamlit run app.py
   ```

6. **Open your browser** and navigate to `http://localhost:8501`

## 🚀 Usage

### Basic Usage

```python
# Example of using the transform_text function
from app import transform_text

message = "FREE prize! Click here to claim your $1000 now!"
processed_message = transform_text(message)
print(processed_message)
# Output: "free prize click claim 1000"
```

### API Integration (Coming Soon)

```python
import requests

# Future API endpoint
response = requests.post(
    "https://api.spam-classifier.com/predict",
    json={"message": "Your message here"}
)
result = response.json()
```

## 📁 Project Structure

```
email-sms-spam-detection/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── nltk.txt              # NLTK data requirements
├── vectorizer1.pkl       # Trained TF-IDF vectorizer
├── svc.pkl              # Trained SVM model
├── README.md            # Project documentation
├── .gitignore           # Git ignore file
│
├── notebooks/           # Jupyter notebooks (if applicable)
│   └── model_training.ipynb
│
├── data/               # Dataset directory (if applicable)
│   ├── spam.csv
│   └── processed/
│
└── models/             # Model artifacts
    ├── vectorizer1.pkl
    └── svc.pkl
```

## 🧠 How It Works

### 1. Text Preprocessing Pipeline

```python
def transform_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Tokenization
    text = nltk.word_tokenize(text)
    
    # Keep only alphanumeric characters
    text = [i for i in text if i.isalnum()]
    
    # Remove stopwords and punctuation
    text = [i for i in text if i not in stopwords.words('english') 
            and i not in string.punctuation]
    
    # Apply stemming
    text = [ps.stem(i) for i in text]
    
    return " ".join(text)
```

### 2. Feature Extraction

- **TF-IDF Vectorization**: Converts text into numerical features
- Captures word importance based on frequency

### 3. Classification

- **Support Vector Machine (SVM)**: Binary classifier
- Trained to distinguish between spam and legitimate messages

## 📊 Model Details

### Performance Metrics

| Metric | Score |
|--------|-------|
| Accuracy | ~97% |
| Precision | ~96% |
| Recall | ~98% |
| F1-Score | ~97% |

### Training Data

- Dataset: SMS Spam Collection Dataset
- Total Messages: 5,572
- Spam Messages: 747 (13.4%)
- Ham Messages: 4,825 (86.6%)

### Model Pipeline

1. **Data Collection**: SMS Spam Collection from UCI ML Repository
2. **Preprocessing**: Text cleaning and normalization
3. **Feature Engineering**: TF-IDF vectorization
4. **Model Selection**: Tested multiple algorithms (Naive Bayes, SVM, Random Forest)
5. **Hyperparameter Tuning**: Grid search for optimal parameters
6. **Model Evaluation**: Cross-validation and test set evaluation

## 🌐 Deployment

### Streamlit Community Cloud

The application is deployed on Streamlit Community Cloud. To deploy your own version:

1. Fork this repository
2. Create account on [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub account
4. Deploy from your forked repository

### Alternative Deployment Options

<details>
<summary><b>Deploy on Heroku</b></summary>

```bash
# Install Heroku CLI
# Create Procfile
echo "web: sh setup.sh && streamlit run app.py" > Procfile

# Create setup.sh
mkdir -p ~/.streamlit/
echo "[server]\nheadless = true\nport = $PORT\nenableCORS = false\n" > ~/.streamlit/config.toml

# Deploy
heroku create your-app-name
git push heroku main
```
</details>

<details>
<summary><b>Deploy with Docker</b></summary>

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

```bash
docker build -t spam-classifier .
docker run -p 8501:8501 spam-classifier
```
</details>

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Steps to Contribute

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Improvement

- [ ] Add multi-language support
- [ ] Implement deep learning models (LSTM, BERT)
- [ ] Add API endpoint for programmatic access
- [ ] Improve UI/UX with custom CSS
- [ ] Add batch prediction feature
- [ ] Implement user feedback system

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Contact

**Dinesh V A**

- GitHub: [@dineshva](https://github.com/yourusername)
- LinkedIn: [Dinesh V A](https://linkedin.com/in/yourusername)
- Email: dinesh.va@example.com

---

<p align="center">
  Made with ❤️ by Dinesh V A
</p>

<p align="center">
  <a href="https://github.com/yourusername/email-sms-spam-detection/issues">Report Bug</a> •
  <a href="https://github.com/yourusername/email-sms-spam-detection/issues">Request Feature</a>
</p>