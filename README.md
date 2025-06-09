# Email/SMS Spam Classifier 📩🚀

A **machine learning-based web application** built with **Streamlit** that detects whether a given text message (Email/SMS) is **Spam** or **Not Spam**.

This project leverages **Natural Language Processing (NLP)** techniques to preprocess text and uses a **Support Vector Machine (SVM)** model trained on SMS data to classify messages. The application provides a simple yet effective **user-friendly interface** for users to check whether their messages contain spam content.

## 🚀 Features

### ✅ **Text Preprocessing**
* Converts text to **lowercase**
* Removes **special characters & punctuation**
* Eliminates **stopwords** (common words that don't add meaning)
* Uses **stemming** (reduces words to their root form)

### ✅ **TF-IDF Vectorization**
* Converts processed text into a numerical format
* Weights words based on their importance

### ✅ **Spam Detection Model**
* Uses a **Support Vector Classifier (SVC)** trained on SMS spam datasets
* Predicts whether a given text is **Spam** or **Not Spam**

### ✅ **User-Friendly UI**
* Simple web interface built with **Streamlit**
* Enter a message and get an instant prediction

## 📂 Project Structure

```
📂 spam-classifier
│── 📄 app.py              # Streamlit app script
│── 📄 vectorizer1.pkl     # Pre-trained TF-IDF vectorizer
│── 📄 svc.pkl             # Trained SVC model
│── 📄 requirements.txt    # Dependencies
│── 📄 README.md           # Project documentation
```

## 🔧 Installation & Setup

### **1️⃣ Clone the Repository**

```sh
git clone https://github.com/your-username/spam-classifier.git
cd spam-classifier
```

### **2️⃣ Install Dependencies**

```sh
pip install -r requirements.txt
```

### **3️⃣ Run the Application**

```sh
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## 🔬 How It Works

1️⃣ **User inputs a message** in the text area
2️⃣ **Text preprocessing** is applied:
   * Converts to lowercase
   * Removes stopwords and punctuation
   * Stems words to their base form
3️⃣ **Vectorization**: Transforms text into numerical data using **TF-IDF**
4️⃣ **Prediction**: The trained **SVM model** classifies the message
5️⃣ **Displays the result**: `Spam` or `Not Spam`

## 📦 Dependencies

* `streamlit`
* `nltk`
* `scikit-learn`
* `pickle`
* `string`

To install them manually:

```sh
pip install streamlit nltk scikit-learn
```

⚠️ **Ensure that the necessary NLTK data is downloaded**

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## 🚀 Deployment

### **🔹 Deploy on Render**

1️⃣ Push your code to **GitHub**
2️⃣ Create a **new Web Service** on Render
3️⃣ Set the **Start Command**:

```sh
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

4️⃣ Deploy & access your app online 🚀

## 📜 Dataset Used

The model is trained on the **SMS Spam Collection Dataset**, which contains:
* **5,574 messages** labeled as `ham` (not spam) or `spam`
* A mix of promotional, phishing, and normal text messages
