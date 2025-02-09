# sentiment_api/urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import ProductViewSet
from .views import analyze_product_sentiment

router = DefaultRouter()
router.register(r"products", ProductViewSet)

urlpatterns = [
    path("", include(router.urls)),
    path("analyze-sentiment/", analyze_product_sentiment, name="analyze-sentiment"),
]
