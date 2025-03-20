# sentiment_api/views.py
from django.http import JsonResponse
from rest_framework.decorators import api_view
from .sentiment_analyzer import SentimentAnalyzer

@api_view(['POST'])
def analyze_product_sentiment(request):
    """
    API endpoint for analyzing sentiment of a product.
    """
    data = request.data
    product_name = data.get("product_name")
    
    if not product_name:
        return JsonResponse({"error": "Product name is required"}, status=400)
    
    sentiment_analyzer = SentimentAnalyzer()
    results = sentiment_analyzer.analyze_sentiment(product_name)
    
    return JsonResponse(results)