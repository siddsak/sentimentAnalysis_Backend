# sentiment_api/sentiment_analyzer.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from scipy.sparse import hstack
import joblib

# Copy all the functions from the original file here:
# train_sentiment_model()
def train_sentiment_model(file_path):
    """
    Train the sentiment analysis model using the review dataset
    
    Parameters:
    file_path (str): Path to the CSV file containing review data
    
    Returns:
    tuple: Returns trained model and vectorizers
    """
    try:
        # Load and preprocess data
        dataset = pd.read_csv(file_path)
        dataset = dataset[['Phone Model', 'Review Text', 'Rating']].dropna()
        
        # Convert ratings to sentiment categories
        dataset['Rating'] = dataset['Rating'].round()
        y_target = dataset['Rating'].map({1: 'Unhappy', 2: 'Unhappy', 3: 'Ok', 4: 'Happy', 5: 'Happy'})
        
        # Feature engineering with corrected parameters
        vectorize_word = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents='unicode',
            analyzer='word',
            stop_words='english',
            max_features=10000
        )
        
        vectorize_char = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents='unicode',
            analyzer='char',
            ngram_range=(2, 6),
            max_features=50000
        )
        
        # Transform the text data
        train_features_word = vectorize_word.fit_transform(dataset['Review Text'])
        train_features_char = vectorize_char.fit_transform(dataset['Review Text'])
        train_features = hstack([train_features_char, train_features_word])
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            train_features, y_target, test_size=0.3, random_state=101, shuffle=True
        )
        
        # Train the model with explicit dual parameter
        model = LinearSVC(class_weight='balanced', dual=True)
        model.fit(X_train, y_train)
        
        # Save the model and vectorizers
        joblib.dump(model, 'sentiment_model.pkl')
        joblib.dump(vectorize_word, 'word_vectorizer.pkl')
        joblib.dump(vectorize_char, 'char_vectorizer.pkl')
        
        return dataset, model, vectorize_word, vectorize_char
    
    except Exception as e:
        print(f"Error in training model: {str(e)}")
        raise

# get_product_sentiment()
def get_product_sentiment(product_name, file_path=None, model=None, word_vectorizer=None, char_vectorizer=None):
    """
    Analyze sentiment for a specific product
    
    Parameters:
    product_name (str): Name of the product to analyze
    file_path (str, optional): Path to the CSV file if model needs to be trained
    model, word_vectorizer, char_vectorizer (optional): Pre-trained model and vectorizers
    
    Returns:
    dict: Sentiment analysis results including review counts and average rating
    """
    try:
        # If model and vectorizers aren't provided, load or train them
        if model is None or word_vectorizer is None or char_vectorizer is None:
            if file_path is None:
                raise ValueError("Either provide trained model and vectorizers or file_path to train new model")
            
            try:
                # Try to load existing model and vectorizers
                model = joblib.load('sentiment_model.pkl')
                word_vectorizer = joblib.load('word_vectorizer.pkl')
                char_vectorizer = joblib.load('char_vectorizer.pkl')
                dataset = pd.read_csv(file_path)
                dataset = dataset[['Phone Model', 'Review Text', 'Rating']].dropna()
            except FileNotFoundError:
                # Train new model if saved model not found
                dataset, model, word_vectorizer, char_vectorizer = train_sentiment_model(file_path)
        
        # Get reviews for the specified product
        product_reviews = dataset[dataset['Phone Model'].str.lower() == product_name.lower()]
        
        if product_reviews.empty:
            return {
                "error": "No reviews found for this product.",
                "Total Reviews": 0,
                "Positive Reviews": 0,
                "Negative Reviews": 0,
                "Neutral Reviews": 0,
                "Average Rating": 0.0
            }
        
        # Prepare features
        review_texts = product_reviews['Review Text']
        review_features = hstack([
            char_vectorizer.transform(review_texts),
            word_vectorizer.transform(review_texts)
        ])
        
        # Get predictions
        predicted_sentiments = model.predict(review_features)
        
        # Calculate metrics
        total_reviews = len(product_reviews)
        positive_count = sum(predicted_sentiments == 'Happy')
        negative_count = sum(predicted_sentiments == 'Unhappy')
        neutral_count = sum(predicted_sentiments == 'Ok')
        avg_rating = product_reviews['Rating'].mean().round(2)
        
        return {
            "Total Reviews": total_reviews,
            "Positive Reviews": positive_count,
            "Negative Reviews": negative_count,
            "Neutral Reviews": neutral_count,
            "Average Rating": avg_rating
        }
    
    except Exception as e:
        return {
            "error": str(e),
            "Total Reviews": 0,
            "Positive Reviews": 0,
            "Negative Reviews": 0,
            "Neutral Reviews": 0,
            "Average Rating": 0.0
        }

# format_sentiment_output()
def format_sentiment_output(product_name, sentiment_result):
    """
    Format the sentiment analysis results in a nice output
    """
    print("\n" + "="*50)
    print(f"Sentiment Analysis Results for {product_name}")
    print("="*50)
    
    if "error" in sentiment_result and sentiment_result["error"] != "":
        print(f"\nError: {sentiment_result['error']}")
    
    print(f"\nTotal Reviews: {sentiment_result['Total Reviews']}")
    print(f"Average Rating: {sentiment_result['Average Rating']}/5.0")
    print("\nSentiment Breakdown:")
    print(f"✓ Positive Reviews: {sentiment_result['Positive Reviews']}")
    print(f"✗ Negative Reviews: {sentiment_result['Negative Reviews']}")
    print(f"○ Neutral Reviews: {sentiment_result['Neutral Reviews']}")
    
    # Calculate percentages if there are reviews
    if sentiment_result['Total Reviews'] > 0:
        pos_percent = (sentiment_result['Positive Reviews'] / sentiment_result['Total Reviews']) * 100
        neg_percent = (sentiment_result['Negative Reviews'] / sentiment_result['Total Reviews']) * 100
        neu_percent = (sentiment_result['Neutral Reviews'] / sentiment_result['Total Reviews']) * 100
        
        print("\nPercentage Breakdown:")
        print(f"Positive: {pos_percent:.1f}%")
        print(f"Negative: {neg_percent:.1f}%")
        print(f"Neutral:  {neu_percent:.1f}%")
    
    print("="*50)
