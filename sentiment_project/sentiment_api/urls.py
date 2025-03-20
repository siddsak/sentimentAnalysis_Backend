# sentiment_api/urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import analyze_product_sentiment

router = DefaultRouter()


urlpatterns = [
    path("", include(router.urls)),
    path('analyze/', analyze_product_sentiment, name='analyze_sentiment'),
]
urlpatterns += router.urls