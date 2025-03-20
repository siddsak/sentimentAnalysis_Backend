# # sentiment_api/sentiment_analyzer.py
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.svm import LinearSVC
# from scipy.sparse import hstack
# import joblib

# # Copy all the functions from the original file here:
# # train_sentiment_model()
# def train_sentiment_model(file_path):

#     """
#     Train the sentiment analysis model using the review dataset

#     Parameters:
#     file_path (str): Path to the CSV file containing review data

#     Returns:
#     tuple: Returns trained model and vectorizers
#     """
#     try:
#         # Load and preprocess data
#         dataset = pd.read_csv(file_path)
#         dataset = dataset[['Phone Model', 'Review Text', 'Rating']].dropna()

#         # Convert ratings to sentiment categories
#         dataset['Rating'] = dataset['Rating'].round()
#         y_target = dataset['Rating'].map({1: 'Unhappy', 2: 'Unhappy', 3: 'Ok', 4: 'Happy', 5: 'Happy'})

#         # Feature engineering with corrected parameters
#         vectorize_word = TfidfVectorizer(
#             sublinear_tf=True,
#             strip_accents='unicode',
#             analyzer='word',
#             stop_words='english',
#             max_features=10000
#         )

#         vectorize_char = TfidfVectorizer(
#             sublinear_tf=True,
#             strip_accents='unicode',
#             analyzer='char',
#             ngram_range=(2, 6),
#             max_features=50000
#         )

#         # Transform the text data
#         train_features_word = vectorize_word.fit_transform(dataset['Review Text'])
#         train_features_char = vectorize_char.fit_transform(dataset['Review Text'])
#         train_features = hstack([train_features_char, train_features_word])

#         # Split the data
#         X_train, X_test, y_train, y_test = train_test_split(
#             train_features, y_target, test_size=0.3, random_state=101, shuffle=True
#         )

#         # Train the model with explicit dual parameter
#         model = LinearSVC(class_weight='balanced', dual=True)
#         model.fit(X_train, y_train)

#         # Save the model and vectorizers
#         joblib.dump(model, 'sentiment_model.pkl')
#         joblib.dump(vectorize_word, 'word_vectorizer.pkl')
#         joblib.dump(vectorize_char, 'char_vectorizer.pkl')

#         return dataset, model, vectorize_word, vectorize_char

#     except Exception as e:
#         print(f"Error in training model: {str(e)}")
#         raise

# # get_product_sentiment()
# def get_product_sentiment(product_name, file_path=None, model=None, word_vectorizer=None, char_vectorizer=None):
#     """
#     Analyze sentiment for a specific product

#     Parameters:
#     product_name (str): Name of the product to analyze
#     file_path (str, optional): Path to the CSV file if model needs to be trained
#     model, word_vectorizer, char_vectorizer (optional): Pre-trained model and vectorizers

#     Returns:
#     dict: Sentiment analysis results including review counts and average rating
#     """
#     try:
#         # If model and vectorizers aren't provided, load or train them
#         if model is None or word_vectorizer is None or char_vectorizer is None:
#             if file_path is None:
#                 raise ValueError("Either provide trained model and vectorizers or file_path to train new model")

#             try:
#                 # Try to load existing model and vectorizers
#                 model = joblib.load('sentiment_model.pkl')
#                 word_vectorizer = joblib.load('word_vectorizer.pkl')
#                 char_vectorizer = joblib.load('char_vectorizer.pkl')
#                 dataset = pd.read_csv(file_path)
#                 dataset = dataset[['Phone Model', 'Review Text', 'Rating']].dropna()
#             except FileNotFoundError:
#                 # Train new model if saved model not found
#                 dataset, model, word_vectorizer, char_vectorizer = train_sentiment_model(file_path)

#         # Get reviews for the specified product
#         product_reviews = dataset[dataset['Phone Model'].str.lower() == product_name.lower()]

#         if product_reviews.empty:
#             return {
#                 "error": "No reviews found for this product.",
#                 "Total Reviews": 0,
#                 "Positive Reviews": 0,
#                 "Negative Reviews": 0,
#                 "Neutral Reviews": 0,
#                 "Average Rating": 0.0
#             }

#         # Prepare features
#         review_texts = product_reviews['Review Text']
#         review_features = hstack([
#             char_vectorizer.transform(review_texts),
#             word_vectorizer.transform(review_texts)
#         ])

#         # Get predictions
#         predicted_sentiments = model.predict(review_features)

#         # Calculate metrics
#         total_reviews = len(product_reviews)
#         positive_count = sum(predicted_sentiments == 'Happy')
#         negative_count = sum(predicted_sentiments == 'Unhappy')
#         neutral_count = sum(predicted_sentiments == 'Ok')
#         avg_rating = product_reviews['Rating'].mean().round(2)

#         return {
#             "Total Reviews": total_reviews,
#             "Positive Reviews": positive_count,
#             "Negative Reviews": negative_count,
#             "Neutral Reviews": neutral_count,
#             "Average Rating": avg_rating
#         }

#     except Exception as e:
#         return {
#             "error": str(e),
#             "Total Reviews": 0,
#             "Positive Reviews": 0,
#             "Negative Reviews": 0,
#             "Neutral Reviews": 0,
#             "Average Rating": 0.0
#         }

# # format_sentiment_output()
# def format_sentiment_output(product_name, sentiment_result):
#     """
#     Format the sentiment analysis results in a nice output
#     """
#     print("\n" + "="*50)
#     print(f"Sentiment Analysis Results for {product_name}")
#     print("="*50)

#     if "error" in sentiment_result and sentiment_result["error"] != "":
#         print(f"\nError: {sentiment_result['error']}")

#     print(f"\nTotal Reviews: {sentiment_result['Total Reviews']}")
#     print(f"Average Rating: {sentiment_result['Average Rating']}/5.0")
#     print("\nSentiment Breakdown:")
#     print(f"✓ Positive Reviews: {sentiment_result['Positive Reviews']}")
#     print(f"✗ Negative Reviews: {sentiment_result['Negative Reviews']}")
#     print(f"○ Neutral Reviews: {sentiment_result['Neutral Reviews']}")

#     # Calculate percentages if there are reviews
#     if sentiment_result['Total Reviews'] > 0:
#         pos_percent = (sentiment_result['Positive Reviews'] / sentiment_result['Total Reviews']) * 100
#         neg_percent = (sentiment_result['Negative Reviews'] / sentiment_result['Total Reviews']) * 100
#         neu_percent = (sentiment_result['Neutral Reviews'] / sentiment_result['Total Reviews']) * 100

#         print("\nPercentage Breakdown:")
#         print(f"Positive: {pos_percent:.1f}%")
#         print(f"Negative: {neg_percent:.1f}%")
#         print(f"Neutral:  {neu_percent:.1f}%")

#     print("="*50)

# sentiment_api/sentiment_analyzer.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.sparse import hstack
import joblib
import numpy as np
from django.conf import settings
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


class SentimentAnalyzer:
    def __init__(self):
        self.model_path = os.path.join(settings.BASE_DIR, "sentiment_api", "models")
        os.makedirs(self.model_path, exist_ok=True)
        
        # Traditional ML model paths
        self.svc_path = os.path.join(self.model_path, "svc_sentiment_model.pkl")
        self.rf_path = os.path.join(self.model_path, "rf_sentiment_model.pkl")
        self.word_vectorizer_path = os.path.join(self.model_path, "word_vectorizer.pkl")
        self.char_vectorizer_path = os.path.join(self.model_path, "char_vectorizer.pkl")
        
        # LSTM model paths
        self.lstm_model_path = os.path.join(self.model_path, "lstm_model.h5")
        self.tokenizer_path = os.path.join(self.model_path, "tokenizer.pkl")

        self.models = None
        self.vectorizers = None
        self.lstm_model = None
        self.tokenizer = None
        self.metrics = None
        self.dataset = None
        self.sentiment_mapping = {1: "Unhappy", 2: "Unhappy", 3: "Ok", 4: "Happy", 5: "Happy"}
        self.reverse_mapping = {"Unhappy": 0, "Ok": 1, "Happy": 2}

    def train_models(self, file_path):
        """Train LinearSVC, RandomForest, and LSTM models"""
        try:
            # Load and preprocess data
            self.dataset = pd.read_csv(file_path)
            self.dataset = self.dataset[
                ["Phone Model", "Review Text", "Rating"]
            ].dropna()

            # Convert ratings to sentiment categories
            self.dataset["Rating"] = self.dataset["Rating"].round()
            y_target = self.dataset["Rating"].map(self.sentiment_mapping)
            
            # Numeric targets for LSTM
            y_lstm = y_target.map(self.reverse_mapping)

            # Feature engineering for traditional ML models
            vectorize_word = TfidfVectorizer(
                sublinear_tf=True,
                strip_accents="unicode",
                analyzer="word",
                stop_words="english",
                max_features=10000,
            )

            vectorize_char = TfidfVectorizer(
                sublinear_tf=True,
                strip_accents="unicode",
                analyzer="char",
                ngram_range=(2, 6),
                max_features=50000,
            )

            # Transform the text data for traditional ML
            train_features_word = vectorize_word.fit_transform(
                self.dataset["Review Text"]
            )
            train_features_char = vectorize_char.fit_transform(
                self.dataset["Review Text"]
            )
            train_features = hstack([train_features_char, train_features_word])

            # Split the data for traditional ML models
            X_train, X_test, y_train, y_test = train_test_split(
                train_features, y_target, test_size=0.3, random_state=101, shuffle=True
            )

            # Train LinearSVC model
            svc_model = LinearSVC(class_weight="balanced", dual=True, max_iter=10000)
            svc_model.fit(X_train, y_train)

            # Train RandomForest model
            rf_model = RandomForestClassifier(
                n_estimators=100, class_weight="balanced", random_state=101, n_jobs=-1
            )
            rf_model.fit(X_train, y_train)

            # Make predictions for traditional ML
            svc_predictions = svc_model.predict(X_test)
            rf_predictions = rf_model.predict(X_test)

            # Store traditional ML models and vectorizers
            self.models = {"svc": svc_model, "rf": rf_model}
            self.vectorizers = {"word": vectorize_word, "char": vectorize_char}

            # Prepare data for LSTM
            tokenizer = Tokenizer(num_words=10000)
            tokenizer.fit_on_texts(self.dataset["Review Text"])
            sequences = tokenizer.texts_to_sequences(self.dataset["Review Text"])
            X_padded = pad_sequences(sequences, maxlen=100)
            
            # Split data for LSTM
            X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(
                X_padded, y_lstm, test_size=0.3, random_state=101, shuffle=True
            )

            # Build LSTM model
            lstm_model = Sequential([
                Embedding(input_dim=10000, output_dim=128, input_length=100),
                LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True),
                LSTM(64, dropout=0.2, recurrent_dropout=0.2),
                Dense(64, activation='relu'),
                Dropout(0.5),
                Dense(3, activation='softmax')  # 3 classes: Unhappy, Ok, Happy
            ])

            # Compile LSTM model
            lstm_model.compile(
                loss='sparse_categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy']
            )

            # Train LSTM model with early stopping
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )
            
            lstm_model.fit(
                X_train_lstm, 
                y_train_lstm,
                epochs=10,
                batch_size=64,
                validation_split=0.2,
                callbacks=[early_stopping]
            )
            
            # Make predictions with LSTM
            lstm_predictions_numeric = lstm_model.predict(X_test_lstm)
            lstm_predictions_classes = np.argmax(lstm_predictions_numeric, axis=1)
            
            # Convert numeric predictions back to text labels
            lstm_predictions = np.array(
                [
                    list(self.reverse_mapping.keys())[list(self.reverse_mapping.values()).index(pred)]
                    for pred in lstm_predictions_classes
                ]
            )
            
            # Store LSTM model and tokenizer
            self.lstm_model = lstm_model
            self.tokenizer = tokenizer
            
            # Calculate and store metrics for all models
            self.metrics = {
                "svc": {
                    "accuracy": float(accuracy_score(y_test, svc_predictions)),
                    "precision": float(
                        precision_score(y_test, svc_predictions, average="weighted")
                    ),
                    "recall": float(
                        recall_score(y_test, svc_predictions, average="weighted")
                    ),
                    "f1_score": float(
                        f1_score(y_test, svc_predictions, average="weighted")
                    ),
                },
                "rf": {
                    "accuracy": float(accuracy_score(y_test, rf_predictions)),
                    "precision": float(
                        precision_score(y_test, rf_predictions, average="weighted")
                    ),
                    "recall": float(
                        recall_score(y_test, rf_predictions, average="weighted")
                    ),
                    "f1_score": float(
                        f1_score(y_test, rf_predictions, average="weighted")
                    ),
                },
                "lstm": {
                    "accuracy": float(accuracy_score(
                        [self.reverse_mapping[label] for label in y_test],
                        lstm_predictions_classes
                    )),
                    "precision": float(
                        precision_score(
                            [self.reverse_mapping[label] for label in y_test],
                            lstm_predictions_classes,
                            average="weighted"
                        )
                    ),
                    "recall": float(
                        recall_score(
                            [self.reverse_mapping[label] for label in y_test],
                            lstm_predictions_classes,
                            average="weighted"
                        )
                    ),
                    "f1_score": float(
                        f1_score(
                            [self.reverse_mapping[label] for label in y_test],
                            lstm_predictions_classes,
                            average="weighted"
                        )
                    ),
                },
            }

            # Save models and vectorizers
            joblib.dump(svc_model, self.svc_path)
            joblib.dump(rf_model, self.rf_path)
            joblib.dump(vectorize_word, self.word_vectorizer_path)
            joblib.dump(vectorize_char, self.char_vectorizer_path)
            
            # Save LSTM model and tokenizer
            lstm_model.save(self.lstm_model_path)
            joblib.dump(tokenizer, self.tokenizer_path)

            return True

        except Exception as e:
            print(f"Error in training models: {str(e)}")
            return False

    def load_models(self):
        """Load pre-trained models, vectorizers, and LSTM model"""
        try:
            # Load traditional ML models
            self.models = {
                "svc": joblib.load(self.svc_path),
                "rf": joblib.load(self.rf_path),
            }
            self.vectorizers = {
                "word": joblib.load(self.word_vectorizer_path),
                "char": joblib.load(self.char_vectorizer_path),
            }
            
            # Load LSTM model and tokenizer
            self.lstm_model = load_model(self.lstm_model_path)
            self.tokenizer = joblib.load(self.tokenizer_path)
            
            return True
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            return False

    def analyze_sentiment(self, product_name, file_path=None):
        """Analyze sentiment for a specific product using all models"""
        try:
            # Load models if not already loaded
            if self.models is None or self.vectorizers is None or self.lstm_model is None:
                if not self.load_models():
                    if file_path is None:
                        return {
                            "error": "Models not found and no training file provided",
                            "status": "error",
                        }
                    if not self.train_models(file_path):
                        return {"error": "Failed to train models", "status": "error"}

            # Load dataset if not already loaded
            if self.dataset is None and file_path:
                self.dataset = pd.read_csv(file_path)
                self.dataset = self.dataset[
                    ["Phone Model", "Review Text", "Rating"]
                ].dropna()

            # Get reviews for the specified product
            product_reviews = self.dataset[
                self.dataset["Phone Model"].str.lower() == product_name.lower()
            ]

            if product_reviews.empty:
                return {
                    "error": "No reviews found for this product",
                    "status": "error",
                    "data": {
                        "total_reviews": 0,
                        "average_rating": 0.0,
                        "svc_results": {"positive": 0, "negative": 0, "neutral": 0},
                        "rf_results": {"positive": 0, "negative": 0, "neutral": 0},
                        "lstm_results": {"positive": 0, "negative": 0, "neutral": 0},
                    },
                }

            # Prepare features for traditional ML models
            review_texts = product_reviews["Review Text"]
            review_features = hstack([
                self.vectorizers["char"].transform(review_texts),
                self.vectorizers["word"].transform(review_texts),
            ])

            # Get predictions from SVC and RF models
            svc_predictions = self.models["svc"].predict(review_features)
            rf_predictions = self.models["rf"].predict(review_features)
            
            # Prepare features for LSTM
            sequences = self.tokenizer.texts_to_sequences(review_texts)
            lstm_input = pad_sequences(sequences, maxlen=100)
            
            # Get LSTM predictions
            lstm_predictions_numeric = self.lstm_model.predict(lstm_input)
            lstm_predictions_classes = np.argmax(lstm_predictions_numeric, axis=1)
            
            # Convert numeric predictions back to text labels
            lstm_predictions = np.array(
                [
                    list(self.reverse_mapping.keys())[list(self.reverse_mapping.values()).index(pred)]
                    for pred in lstm_predictions_classes
                ]
            )

            # Calculate metrics
            results = {
                "status": "success",
                "data": {
                    "total_reviews": len(product_reviews),
                    "average_rating": float(product_reviews["Rating"].mean().round(2)),
                    "svc_results": {
                        "positive": int(sum(svc_predictions == "Happy")),
                        "negative": int(sum(svc_predictions == "Unhappy")),
                        "neutral": int(sum(svc_predictions == "Ok")),
                    },
                    "rf_results": {
                        "positive": int(sum(rf_predictions == "Happy")),
                        "negative": int(sum(rf_predictions == "Unhappy")),
                        "neutral": int(sum(rf_predictions == "Ok")),
                    },
                    "lstm_results": {
                        "positive": int(sum(lstm_predictions == "Happy")),
                        "negative": int(sum(lstm_predictions == "Unhappy")),
                        "neutral": int(sum(lstm_predictions == "Ok")),
                    },
                },
            }

            # Add model performance metrics if available
            if self.metrics:
                results["data"]["model_metrics"] = self.metrics

            return results

        except Exception as e:
            return {
                "error": str(e),
                "status": "error",
                "data": {
                    "total_reviews": 0,
                    "average_rating": 0.0,
                    "svc_results": {"positive": 0, "negative": 0, "neutral": 0},
                    "rf_results": {"positive": 0, "negative": 0, "neutral": 0},
                    "lstm_results": {"positive": 0, "negative": 0, "neutral": 0},
                },
            }

    def get_available_models(self):
        """Get list of unique phone models in the dataset"""
        if self.dataset is not None:
            return self.dataset['Phone Model'].unique()
        return []