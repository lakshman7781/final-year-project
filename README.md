# Fake News Detection System 

## Project Aim
The goal of this project is to combat the spread of misinformation by providing a robust, multi-faceted tool for verifying news credibility. Beyond simple classification, it addresses the **psychological impact** of fake news by offering:
1.  **Credibility Analysis**: Using Machine Learning to detect fake news.
2.  **Live Fact-Checking**: verifying headlines against reputable sources (Google News).
3.  **Stress Reduction**: Providing resources if the content is alarming or fake.
4.  **Reputation Management**: Helping individuals/organizations manage damage from fake stories.

## Key Features
- **ML Detection**: Logistic Regression model trained on 40,000+ articles (ISOT Dataset).
- **Sentiment Analysis**: Detects emotional manipulation and "trigger words" (e.g., "panic", "secret").
- **Analytics Dashboard**: Tracks your analysis history and visualizes Real vs. Fake trends.
- **Dynamic Reports**: Generates detailed content analysis reports.

## How it Works
1.  **Input**: User enters a news headline or article.
2.  **Preprocessing**: Text is cleaned (stop words removed, lowercasing).
3.  **Vectorization**: Converted to numerical features using TF-IDF.
4.  **Prediction**: The trained model classifies it as Real or Fake.
5.  **Enhancement**:
    - **Scraper**: Checks Google News for matching stories.
    - **NLP**: TextBlob extracts sentiment and entities (e.g., "Hasina").

## Installation & Setup

### Prerequisites
- Python 3.9+
- Pip (Python Package Manager)

### Local Setup
1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/lakshman7781/final-year-project.git
    cd final-year-project
    ```
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Train the Model**:
    ```bash
    python3 ml_engine/train_model.py
    ```
4.  **Run the App**:
    ```bash
    python3 backend/app.py
    ```
5.  **Open in Browser**: Go to `http://127.0.0.1:5001`

### AWS Deployment (VM)
1.  **Connect to VM**: `ssh -i key.pem user@vm-ip`
2.  **Run Setup Script**:
    ```bash
    git clone https://github.com/lakshman7781/final-year-project.git
    cd final-year-project
    chmod +x setup_aws.sh
    ./setup_aws.sh
    ```
3.  **Start App**:
    ```bash
    source venv/bin/activate
    python3 ml_engine/train_model.py
    python3 backend/app.py
    ```
