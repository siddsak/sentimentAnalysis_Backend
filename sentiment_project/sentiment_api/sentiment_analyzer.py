
# # sentiment_api/sentiment_analyzer.py
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.svm import LinearSVC
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from scipy.sparse import hstack
# import joblib
# import numpy as np
# from django.conf import settings
# import os
# import tensorflow as tf
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# from keras.models import Sequential, load_model
# from keras.layers import Embedding, LSTM, Dense, Dropout
# from keras.callbacks import EarlyStopping


# class SentimentAnalyzer:
#     def __init__(self):
#         self.model_path = os.path.join(settings.BASE_DIR, "sentiment_api", "models")
#         os.makedirs(self.model_path, exist_ok=True)
        
#         # Traditional ML model paths
#         self.svc_path = os.path.join(self.model_path, "svc_sentiment_model.pkl")
#         self.rf_path = os.path.join(self.model_path, "rf_sentiment_model.pkl")
#         self.word_vectorizer_path = os.path.join(self.model_path, "word_vectorizer.pkl")
#         self.char_vectorizer_path = os.path.join(self.model_path, "char_vectorizer.pkl")
        
#         # LSTM model paths
#         self.lstm_model_path = os.path.join(self.model_path, "lstm_model.h5")
#         self.tokenizer_path = os.path.join(self.model_path, "tokenizer.pkl")

#         self.models = None
#         self.vectorizers = None
#         self.lstm_model = None
#         self.tokenizer = None
#         self.metrics = None
#         self.dataset = None
#         self.sentiment_mapping = {1: "Unhappy", 2: "Unhappy", 3: "Ok", 4: "Happy", 5: "Happy"}
#         self.reverse_mapping = {"Unhappy": 0, "Ok": 1, "Happy": 2}

#     def train_models(self, file_path):
#         """Train LinearSVC, RandomForest, and LSTM models"""
#         try:
#             # Load and preprocess data
#             self.dataset = pd.read_csv("phone_reviews.csv")
#             self.dataset = self.dataset[
#                 ["Phone Model", "Review Text", "Rating"]
#             ].dropna()

#             # Convert ratings to sentiment categories
#             self.dataset["Rating"] = self.dataset["Rating"].round()
#             y_target = self.dataset["Rating"].map(self.sentiment_mapping)
            
#             # Numeric targets for LSTM
#             y_lstm = y_target.map(self.reverse_mapping)

#             # Feature engineering for traditional ML models
#             vectorize_word = TfidfVectorizer(
#                 sublinear_tf=True,
#                 strip_accents="unicode",
#                 analyzer="word",
#                 stop_words="english",
#                 max_features=10000,
#             )

#             vectorize_char = TfidfVectorizer(
#                 sublinear_tf=True,
#                 strip_accents="unicode",
#                 analyzer="char",
#                 ngram_range=(2, 6),
#                 max_features=50000,
#             )

#             # Transform the text data for traditional ML
#             train_features_word = vectorize_word.fit_transform(
#                 self.dataset["Review Text"]
#             )
#             train_features_char = vectorize_char.fit_transform(
#                 self.dataset["Review Text"]
#             )
#             train_features = hstack([train_features_char, train_features_word])

#             # Split the data for traditional ML models
#             X_train, X_test, y_train, y_test = train_test_split(
#                 train_features, y_target, test_size=0.3, random_state=101, shuffle=True
#             )

#             # Train LinearSVC model
#             svc_model = LinearSVC(class_weight="balanced", dual=True, max_iter=10000)
#             svc_model.fit(X_train, y_train)

#             # Train RandomForest model
#             rf_model = RandomForestClassifier(
#                 n_estimators=100, class_weight="balanced", random_state=101, n_jobs=-1
#             )
#             rf_model.fit(X_train, y_train)

#             # Make predictions for traditional ML
#             svc_predictions = svc_model.predict(X_test)
#             rf_predictions = rf_model.predict(X_test)

#             # Store traditional ML models and vectorizers
#             self.models = {"svc": svc_model, "rf": rf_model}
#             self.vectorizers = {"word": vectorize_word, "char": vectorize_char}

#             # Prepare data for LSTM
#             tokenizer = Tokenizer(num_words=10000)
#             tokenizer.fit_on_texts(self.dataset["Review Text"])
#             sequences = tokenizer.texts_to_sequences(self.dataset["Review Text"])
#             X_padded = pad_sequences(sequences, maxlen=100)
            
#             # Split data for LSTM
#             X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(
#                 X_padded, y_lstm, test_size=0.3, random_state=101, shuffle=True
#             )

#             # Build LSTM model
#             lstm_model = Sequential([
#                 Embedding(input_dim=10000, output_dim=128, input_length=100),
#                 LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True),
#                 LSTM(64, dropout=0.2, recurrent_dropout=0.2),
#                 Dense(64, activation='relu'),
#                 Dropout(0.5),
#                 Dense(3, activation='softmax')  # 3 classes: Unhappy, Ok, Happy
#             ])

#             # Compile LSTM model
#             lstm_model.compile(
#                 loss='sparse_categorical_crossentropy',
#                 optimizer='adam',
#                 metrics=['accuracy']
#             )

#             # Train LSTM model with early stopping
#             early_stopping = EarlyStopping(
#                 monitor='val_loss',
#                 patience=3,
#                 restore_best_weights=True
#             )
            
#             lstm_model.fit(
#                 X_train_lstm, 
#                 y_train_lstm,
#                 epochs=10,
#                 batch_size=64,
#                 validation_split=0.2,
#                 callbacks=[early_stopping]
#             )
            
#             # Make predictions with LSTM
#             lstm_predictions_numeric = lstm_model.predict(X_test_lstm)
#             lstm_predictions_classes = np.argmax(lstm_predictions_numeric, axis=1)
            
#             # Convert numeric predictions back to text labels
#             lstm_predictions = np.array(
#                 [
#                     list(self.reverse_mapping.keys())[list(self.reverse_mapping.values()).index(pred)]
#                     for pred in lstm_predictions_classes
#                 ]
#             )
            
#             # Store LSTM model and tokenizer
#             self.lstm_model = lstm_model
#             self.tokenizer = tokenizer
            
#             # Calculate and store metrics for all models
#             self.metrics = {
#                 "svc": {
#                     "accuracy": float(accuracy_score(y_test, svc_predictions)),
#                     "precision": float(
#                         precision_score(y_test, svc_predictions, average="weighted")
#                     ),
#                     "recall": float(
#                         recall_score(y_test, svc_predictions, average="weighted")
#                     ),
#                     "f1_score": float(
#                         f1_score(y_test, svc_predictions, average="weighted")
#                     ),
#                 },
#                 "rf": {
#                     "accuracy": float(accuracy_score(y_test, rf_predictions)),
#                     "precision": float(
#                         precision_score(y_test, rf_predictions, average="weighted")
#                     ),
#                     "recall": float(
#                         recall_score(y_test, rf_predictions, average="weighted")
#                     ),
#                     "f1_score": float(
#                         f1_score(y_test, rf_predictions, average="weighted")
#                     ),
#                 },
#                 "lstm": {
#                     "accuracy": float(accuracy_score(
#                         [self.reverse_mapping[label] for label in y_test],
#                         lstm_predictions_classes
#                     )),
#                     "precision": float(
#                         precision_score(
#                             [self.reverse_mapping[label] for label in y_test],
#                             lstm_predictions_classes,
#                             average="weighted"
#                         )
#                     ),
#                     "recall": float(
#                         recall_score(
#                             [self.reverse_mapping[label] for label in y_test],
#                             lstm_predictions_classes,
#                             average="weighted"
#                         )
#                     ),
#                     "f1_score": float(
#                         f1_score(
#                             [self.reverse_mapping[label] for label in y_test],
#                             lstm_predictions_classes,
#                             average="weighted"
#                         )
#                     ),
#                 },
#             }

#             # Save models and vectorizers
#             joblib.dump(svc_model, self.svc_path)
#             joblib.dump(rf_model, self.rf_path)
#             joblib.dump(vectorize_word, self.word_vectorizer_path)
#             joblib.dump(vectorize_char, self.char_vectorizer_path)
            
#             # Save LSTM model and tokenizer
#             lstm_model.save(self.lstm_model_path)
#             joblib.dump(tokenizer, self.tokenizer_path)

#             return True

#         except Exception as e:
#             print(f"Error in training models: {str(e)}")
#             return False

#     def load_models(self):
#         """Load pre-trained models, vectorizers, and LSTM model"""
#         try:
#             # Load traditional ML models
#             self.models = {
#                 "svc": joblib.load(self.svc_path),
#                 "rf": joblib.load(self.rf_path),
#             }
#             self.vectorizers = {
#                 "word": joblib.load(self.word_vectorizer_path),
#                 "char": joblib.load(self.char_vectorizer_path),
#             }
            
#             # Load LSTM model and tokenizer
#             self.lstm_model = load_model(self.lstm_model_path)
#             self.tokenizer = joblib.load(self.tokenizer_path)
            
#             return True
#         except Exception as e:
#             print(f"Error loading models: {str(e)}")
#             return False

#     def analyze_sentiment(self, product_name, file_path="phone_reviews.csv"):
#         """Analyze sentiment for a specific product using all models"""
#         print(f"Dataset loaded: {self.dataset is not None}")
#         print(f"Vectorizers loaded: {self.vectorizers is not None}")
#         print(f"Models loaded: {self.models is not None}")
#         print(f"Tokenizer loaded: {self.tokenizer is not None}")    
#         print(f"Reverse Mapping: {self.reverse_mapping}")

#         try:
#             # Load models if not already loaded
#             if self.models is None or self.vectorizers is None or self.lstm_model is None:
#                 if not self.load_models():
#                     if file_path is None:
#                         return {
#                             "error": "Models not found and no training file provided",
#                             "status": "error",
#                         }
#                     if not self.train_models(file_path):
#                         return {"error": "Failed to train models", "status": "error"}

#             # Load dataset if not already loaded
#             if self.dataset is None and file_path:
#                 self.dataset = pd.read_csv(file_path)
#                 self.dataset = self.dataset[
#                     ["Phone Model", "Review Text", "Rating"]
#                 ].dropna()

#             # Get reviews for the specified product
#             product_reviews = self.dataset[
#                 self.dataset["Phone Model"].str.lower() == product_name.lower()
#             ]

#             if product_reviews.empty:
#                 return {
#                     "error": "No reviews found for this product",
#                     "status": "error",
#                     "data": {
#                         "total_reviews": 0,
#                         "average_rating": 0.0,
#                         "svc_results": {"positive": 0, "negative": 0, "neutral": 0},
#                         "rf_results": {"positive": 0, "negative": 0, "neutral": 0},
#                         "lstm_results": {"positive": 0, "negative": 0, "neutral": 0},
#                     },
#                 }

#             # Prepare features for traditional ML models
#             review_texts = product_reviews["Review Text"]
#             review_features = hstack([
#                 self.vectorizers["char"].transform(review_texts),
#                 self.vectorizers["word"].transform(review_texts),
#             ])

#             # Get predictions from SVC and RF models
#             svc_predictions = self.models["svc"].predict(review_features)
#             rf_predictions = self.models["rf"].predict(review_features)
            
#             # Prepare features for LSTM
#             sequences = self.tokenizer.texts_to_sequences(review_texts)
#             lstm_input = pad_sequences(sequences, maxlen=100)
            
#             # Get LSTM predictions
#             lstm_predictions_numeric = self.lstm_model.predict(lstm_input)
#             lstm_predictions_classes = np.argmax(lstm_predictions_numeric, axis=1)
            
#             # Convert numeric predictions back to text labels
#             lstm_predictions = np.array(
#                 [
#                     list(self.reverse_mapping.keys())[list(self.reverse_mapping.values()).index(pred)]
#                     for pred in lstm_predictions_classes
#                 ]
#             )

#             # Calculate metrics
#             results = {
#                 "status": "success",
#                 "data": {
#                     "total_reviews": len(product_reviews),
#                     "average_rating": float(product_reviews["Rating"].mean().round(2)),
#                     "svc_results": {
#                         "positive": int(sum(svc_predictions == "Happy")),
#                         "negative": int(sum(svc_predictions == "Unhappy")),
#                         "neutral": int(sum(svc_predictions == "Ok")),
#                     },
#                     "rf_results": {
#                         "positive": int(sum(rf_predictions == "Happy")),
#                         "negative": int(sum(rf_predictions == "Unhappy")),
#                         "neutral": int(sum(rf_predictions == "Ok")),
#                     },
#                     "lstm_results": {
#                         "positive": int(sum(lstm_predictions == "Happy")),
#                         "negative": int(sum(lstm_predictions == "Unhappy")),
#                         "neutral": int(sum(lstm_predictions == "Ok")),
#                     },
#                 },
#             }

#             # Add model performance metrics if available
#             if self.metrics:
#                 results["data"]["model_metrics"] = self.metrics
#             return results

#         except Exception as e:
#             return {
#                 "error": str(e),
#                 "status": "error",
#                 "data": {
#                     "total_reviews": 0,
#                     "average_rating": 0.0,
#                     "svc_results": {"positive": 0, "negative": 0, "neutral": 0},
#                     "rf_results": {"positive": 0, "negative": 0, "neutral": 0},
#                     "lstm_results": {"positive": 0, "negative": 0, "neutral": 0},
#                 },
#             }

#     def get_available_models(self):
#         """Get list of unique phone models in the dataset"""
#         if self.dataset is not None:
#             return self.dataset['Phone Model'].unique()
#         return []


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
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping


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
        
        # Metrics path - add this to store and retrieve metrics
        self.metrics_path = os.path.join(self.model_path, "model_metrics.pkl")

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
            self.dataset = pd.read_csv("phone_reviews.csv")
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
            
            # Save metrics to a file so they can be loaded later
            joblib.dump(self.metrics, self.metrics_path)

            return True

        except Exception as e:
            print(f"Error in training models: {str(e)}")
            return False

    def load_models(self):
        """Load pre-trained models, vectorizers, LSTM model, and metrics"""
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
            
            # Load metrics if they exist
            if os.path.exists(self.metrics_path):
                self.metrics = joblib.load(self.metrics_path)
                print("Metrics loaded successfully:", self.metrics)
            else:
                print("Metrics file not found")
            
            return True
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            return False
            
    def generate_metrics(self, file_path="phone_reviews.csv"):
        """Generate and save metrics for existing models using test data"""
        try:
            # Check if models are loaded
            if self.models is None or self.vectorizers is None or self.lstm_model is None:
                if not self.load_models():
                    return False
                    
            # Load dataset if not already loaded
            if self.dataset is None and file_path:
                self.dataset = pd.read_csv(file_path)
                self.dataset = self.dataset[
                    ["Phone Model", "Review Text", "Rating"]
                ].dropna()
                
            if self.dataset is None:
                print("Cannot generate metrics: dataset not available")
                return False
                
            # Prepare data for metrics calculation
            self.dataset["Rating"] = self.dataset["Rating"].round()
            y_target = self.dataset["Rating"].map(self.sentiment_mapping)
            y_lstm = y_target.map(self.reverse_mapping)
            
            # Feature engineering
            train_features_word = self.vectorizers["word"].transform(self.dataset["Review Text"])
            train_features_char = self.vectorizers["char"].transform(self.dataset["Review Text"])
            train_features = hstack([train_features_char, train_features_word])
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                train_features, y_target, test_size=0.3, random_state=101, shuffle=True
            )
            
            # Make predictions
            svc_predictions = self.models["svc"].predict(X_test)
            rf_predictions = self.models["rf"].predict(X_test)
            
            # Prepare LSTM data
            sequences = self.tokenizer.texts_to_sequences(self.dataset["Review Text"])
            X_padded = pad_sequences(sequences, maxlen=100)
            
            # Split LSTM data
            X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(
                X_padded, y_lstm, test_size=0.3, random_state=101, shuffle=True
            )
            
            # Make LSTM predictions
            lstm_predictions_numeric = self.lstm_model.predict(X_test_lstm)
            lstm_predictions_classes = np.argmax(lstm_predictions_numeric, axis=1)
            
            # Calculate and store metrics
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
                        y_test_lstm, lstm_predictions_classes
                    )),
                    "precision": float(
                        precision_score(
                            y_test_lstm, lstm_predictions_classes, average="weighted"
                        )
                    ),
                    "recall": float(
                        recall_score(
                            y_test_lstm, lstm_predictions_classes, average="weighted"
                        )
                    ),
                    "f1_score": float(
                        f1_score(
                            y_test_lstm, lstm_predictions_classes, average="weighted"
                        )
                    ),
                },
            }
            
            # Save metrics
            joblib.dump(self.metrics, self.metrics_path)
            print("Metrics generated and saved successfully:", self.metrics)
            return True
            
        except Exception as e:
            print(f"Error generating metrics: {str(e)}")
            return False

    def analyze_sentiment(self, product_name, file_path="phone_reviews.csv"):
        """Analyze sentiment for a specific product using all models"""
        print(f"Dataset loaded: {self.dataset is not None}")
        print(f"Vectorizers loaded: {self.vectorizers is not None}")
        print(f"Models loaded: {self.models is not None}")
        print(f"Tokenizer loaded: {self.tokenizer is not None}")    
        print(f"Metrics loaded: {self.metrics is not None}")
        print(f"Reverse Mapping: {self.reverse_mapping}")

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

            # Check if metrics are available, generate if not
            if self.metrics is None:
                print("Metrics not found, generating now...")
                self.generate_metrics(file_path)

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

            # Create response with the base results
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

            # Add model performance metrics if they exist
            if self.metrics is not None:
                print("Including metrics in response:", self.metrics)
                results["data"]["model_metrics"] = self.metrics
            
            return results

        except Exception as e:
            print(f"Error in analyze_sentiment: {str(e)}")
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