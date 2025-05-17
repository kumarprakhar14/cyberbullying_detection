# Cyberbullying Detection Using Machine Learning

A machine learning-based web application that detects cyberbullying content in text using Natural Language Processing (NLP) and Support Vector Classification.

## Overview

This project implements a cyberbullying detection system that:
- Analyzes text input for potential cyberbullying content
- Uses TF-IDF vectorization for text feature extraction
- Employs multiple ML algorithms with Linear SVC showing best performance
- Provides a user-friendly web interface for real-time detection

## Project Structure

```
cyberbullying_detection/
├── app.py                         # Flask web application
├── dataset.csv                    # Training dataset
├── stopwords.txt                  # Custom stopwords list
├── templates/
│   └── index.html                # Web interface template
├── LinearSVCTuned.pkl            # Trained model
├── tfidfvectoizer.pkl            # TF-IDF vectorizer
└── Cyber Bulling Detection Using Python.ipynb  # Development notebook
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kumarprakhar14/cyberbullying_detection.git
cd cyberbullying_detection
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

- Python 3.7+
- Flask
- NumPy
- Pandas
- Scikit-learn
- NLTK
- XGBoost
- Seaborn
- Matplotlib

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open a web browser and navigate to:
```
http://localhost:5000
```

3. Enter text in the provided input field and click "Analyze Text" to detect cyberbullying content.

## Model Details

### Data Preprocessing
- Text cleaning and normalization
- Stopword removal using custom stopwords list
- Lemmatization using NLTK
- TF-IDF vectorization

### Machine Learning Models Evaluated
- Linear Support Vector Classification (LinearSVC)
- Logistic Regression
- Multinomial Naive Bayes
- Decision Tree Classifier
- AdaBoost Classifier
- Bagging Classifier
- SGD Classifier

### Model Performance
- Best performing model: LinearSVC
- Accuracy: ~92%
- F1 Score: ~91%
- Precision: ~90%
- Recall: ~92%

## Training the Model

To retrain the model using the Jupyter notebook:

1. Open `Cyber Bulling Detection Using Python.ipynb`
2. Run all cells sequentially
3. The trained model will be saved as `LinearSVCTuned.pkl`

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Kumar Prakhar
- GitHub: [kumarprakhar14](https://github.com/kumarprakhar14)
- Instagram: [@kumarprakharkp143](https://www.instagram.com/kumarprakharkp143/)

## Acknowledgments

- Dataset contributors
- Open-source ML community
- Flask framework developers
