This project demonstrates the development of an email spam detection system using machine learning techniques. The system uses the popular Enron Spam Dataset to train a Logistic Regression model that classifies emails as spam or not spam (ham) based on their content.

To make the model accessible and user-friendly, a web application interface is created using Streamlit, allowing users to input email text and receive real-time predictions on whether the email is spam.<br>
Features
Data preprocessing and model training: Uses TF-IDF vectorization and Logistic Regression to build an effective text classification model.

High accuracy: Achieves around 98% accuracy on the test set.

Interactive GUI: Streamlit app to input any email text and predict spam or ham instantly.

Model persistence: Saves trained model and vectorizer objects using joblib for easy reuse.

Modular code: Separate scripts for training (main.py) and the web app (app.py).<br>
Dataset
The project uses the Enron Spam Dataset, which contains thousands of emails labeled as spam or ham.
Note: Due to size and licensing restrictions, the dataset is not included in this repository.
You can download it from Kaggle<br>
Project Structure:<br>
├── data/
│   └── enron_spam_data.csv    # Place your dataset here (not included)<br>
├── app.py                    # Streamlit app to predict spam<br>
├── train_model.py            # Script to train and save the model<br>
├── spam_model.pkl            # Saved trained model (not included)<br>
├── vectorizer.pkl            # Saved vectorizer (not included)<br>
├── requirements.txt          # Required Python packages<br>
└── README.md                 # Project documentation<br>
