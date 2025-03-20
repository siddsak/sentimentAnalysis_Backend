
# # sentiment_api/views.py
# from rest_framework import viewsets, status
# from rest_framework.decorators import api_view
# from rest_framework.response import Response
# from .sentiment_analyzer import get_product_sentiment
# import pandas as pd
# from rest_framework.decorators import api_view
# from rest_framework import viewsets, status
# from rest_framework.decorators import action
# from rest_framework.response import Response
# from .models import Product, SentimentResult
# from .serializers import ProductSerializer, SentimentResultSerializer
# from .sentiment_analyzer import get_product_sentiment

# class ProductViewSet(viewsets.ModelViewSet):
#     queryset = Product.objects.all()
#     serializer_class = ProductSerializer

#     @action(detail=True, methods=['post'])
#     def analyze_sentiment(self, request, pk=None):
#         product = self.get_object()
        
#         try:
#             # Get sentiment analysis results
#             results = get_product_sentiment(
#                 product.name,
#                 file_path='phone_reviews.csv'  # You might want to make this configurable
#             )
            
#             # Save results to database
#             sentiment_result = SentimentResult.objects.create(
#                 product=product,
#                 **results
#             )
            
#             serializer = SentimentResultSerializer(sentiment_result)
#             return Response(serializer.data)
#         except Exception as e:
#             return Response(
#                 {'error': str(e)},
#                 status=status.HTTP_500_INTERNAL_SERVER_ERROR
#             )

#     @action(detail=True, methods=['get'])
#     def sentiment_history(self, request, pk=None):
#         product = self.get_object()
#         sentiment_results = product.sentiment_results.all().order_by('-created_at')
#         serializer = SentimentResultSerializer(sentiment_results, many=True)
#         return Response(serializer.data)
    
# # sentiment_api/views.py
# @api_view(['POST'])
# def analyze_product_sentiment(request):
#     """
#     Endpoint to get sentiment analysis for a product
#     Request body should contain: {"product_name": "product_name"}
#     """
#     try:
#         product_name = request.data.get('product_name')
        
#         if not product_name:
#             return Response(
#                 {'error': 'Product name is required'}, 
#                 status=status.HTTP_400_BAD_REQUEST
#             )

#         # Read the CSV file
#         df = pd.read_csv('phone_reviews.csv')
        
#         # Print unique phone models for debugging
#         print("Available phone models:", df['Phone Model'].unique())
        
#         # Case-insensitive partial matching for product name
#         product_reviews = df[df['Phone Model'].str.contains(product_name, case=False, na=False)]
        
#         if product_reviews.empty:
#             return Response({
#                 'error': f'No reviews found for product containing "{product_name}"',
#                 'available_products': df['Phone Model'].unique().tolist()
#             }, status=status.HTTP_404_NOT_FOUND)
        
#         # Calculate metrics
#         total_reviews = len(product_reviews)
#         average_rating = round(product_reviews['Rating'].mean(), 2)
        
#         # Calculate sentiment breakdown
#         positive_reviews = len(product_reviews[product_reviews['Rating'] >= 4])
#         negative_reviews = len(product_reviews[product_reviews['Rating'] <= 2])
#         neutral_reviews = len(product_reviews[(product_reviews['Rating'] > 2) & (product_reviews['Rating'] < 4)])
        
#         response_data = {
#             'product_name': product_name,
#             'sentiment_analysis': {
#                 'total_reviews': total_reviews,
#                 'average_rating': average_rating,
#                 'sentiment_breakdown': {
#                     'positive_reviews': positive_reviews,
#                     'negative_reviews': negative_reviews,
#                     'neutral_reviews': neutral_reviews
#                 }
#             },
#             'matched_products': product_reviews['Phone Model'].unique().tolist()
#         }
        
#         return Response(response_data, status=status.HTTP_200_OK)
        
#     except Exception as e:
#         import traceback
#         print("Error:", str(e))
#         print("Traceback:", traceback.format_exc())
#         return Response(
#             {'error': str(e)},
#             status=status.HTTP_500_INTERNAL_SERVER_ERROR
#         )

# sentiment_api/views.py
# sentiment_api/views.py
from rest_framework import viewsets, status
from rest_framework.decorators import api_view, action
from rest_framework.response import Response
from .models import Product, SentimentResult
from .serializers import ProductSerializer, SentimentResultSerializer
import pandas as pd
from .sentiment_analyzer import SentimentAnalyzer

class ProductViewSet(viewsets.ModelViewSet):
    queryset = Product.objects.all()
    serializer_class = ProductSerializer

    @action(detail=True, methods=['post'])
    def analyze_sentiment(self, request, pk=None):
        product = self.get_object()
        
        try:
            # Initialize the sentiment analyzer
            analyzer = SentimentAnalyzer()
            
            # Get sentiment analysis results with all models
            results = analyzer.analyze_sentiment(
                product.name,
                file_path='phone_reviews.csv'
            )
            
            if results['status'] == 'error':
                return Response(
                    {'error': results['error']},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
            
            # Transform results to match your model structure
            # Use ensemble approach (taking max across all models)
            sentiment_data = {
                'total_reviews': results['data']['total_reviews'],
                'average_rating': results['data']['average_rating'],
                'positive_reviews': max(
                    results['data']['svc_results']['positive'],
                    results['data']['rf_results']['positive'],
                    results['data']['lstm_results']['positive']
                ),
                'negative_reviews': max(
                    results['data']['svc_results']['negative'],
                    results['data']['rf_results']['negative'],
                    results['data']['lstm_results']['negative']
                ),
                'neutral_reviews': max(
                    results['data']['svc_results']['neutral'],
                    results['data']['rf_results']['neutral'],
                    results['data']['lstm_results']['neutral']
                ),
                'model_comparison': results['data'].get('model_metrics', {})
            }
            
            # Save results to database
            sentiment_result = SentimentResult.objects.create(
                product=product,
                **sentiment_data
            )
            
            serializer = SentimentResultSerializer(sentiment_result)
            return Response(serializer.data)
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=True, methods=['get'])
    def sentiment_history(self, request, pk=None):
        product = self.get_object()
        sentiment_results = product.sentiment_results.all().order_by('-created_at')
        serializer = SentimentResultSerializer(sentiment_results, many=True)
        return Response(serializer.data)

@api_view(['POST'])
def analyze_product_sentiment(request):
    """
    Enhanced endpoint to get sentiment analysis for a product using LinearSVC, RandomForest, and LSTM models
    Request body should contain: {"product_name": "product_name"}
    """
    try:
        product_name = request.data.get('product_name')
        
        if not product_name:
            return Response(
                {'error': 'Product name is required'}, 
                status=status.HTTP_400_BAD_REQUEST
            )

        # Initialize the sentiment analyzer
        analyzer = SentimentAnalyzer()
        
        # Get available products
        available_products = analyzer.get_available_models()
        print("Available phone models:", available_products)
        
        # Perform sentiment analysis with all models
        results = analyzer.analyze_sentiment(
            product_name,
            file_path='phone_reviews.csv'
        )
        
        if results['status'] == 'error':
            if 'No reviews found' in results['error']:
                return Response({
                    'error': f'No reviews found for product containing "{product_name}"',
                    'available_products': available_products.tolist()
                }, status=status.HTTP_404_NOT_FOUND)
            return Response(
                {'error': results['error']},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
        # Prepare enhanced response with all models comparison
        response_data = {
            'product_name': product_name,
            'sentiment_analysis': {
                'total_reviews': results['data']['total_reviews'],
                'average_rating': results['data']['average_rating'],
                'svc_results': {
                    'positive_reviews': results['data']['svc_results']['positive'],
                    'negative_reviews': results['data']['svc_results']['negative'],
                    'neutral_reviews': results['data']['svc_results']['neutral']
                },
                'random_forest_results': {
                    'positive_reviews': results['data']['rf_results']['positive'],
                    'negative_reviews': results['data']['rf_results']['negative'],
                    'neutral_reviews': results['data']['rf_results']['neutral']
                },
                'lstm_results': {
                    'positive_reviews': results['data']['lstm_results']['positive'],
                    'negative_reviews': results['data']['lstm_results']['negative'],
                    'neutral_reviews': results['data']['lstm_results']['neutral']
                },
                'ensemble_results': {
                    'positive_reviews': max(
                        results['data']['svc_results']['positive'],
                        results['data']['rf_results']['positive'],
                        results['data']['lstm_results']['positive']
                    ),
                    'negative_reviews': max(
                        results['data']['svc_results']['negative'],
                        results['data']['rf_results']['negative'],
                        results['data']['lstm_results']['negative']
                    ),
                    'neutral_reviews': max(
                        results['data']['svc_results']['neutral'],
                        results['data']['rf_results']['neutral'],
                        results['data']['lstm_results']['neutral']
                    )
                }
            },
            'model_metrics': results['data'].get('model_metrics', {}),
            'matched_products': [product_name]
        }
        
        return Response(response_data, status=status.HTTP_200_OK)
        
    except Exception as e:
        import traceback
        print("Error:", str(e))
        print("Traceback:", traceback.format_exc())
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )