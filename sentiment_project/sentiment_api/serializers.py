#serializer.py
from rest_framework import serializers
from .models import Product, SentimentResult


class SentimentResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = SentimentResult
        fields = [
            "id",
            "total_reviews",
            "positive_reviews",
            "negative_reviews",
            "neutral_reviews",
            "average_rating",
            "created_at",
        ]


class ProductSerializer(serializers.ModelSerializer):
    sentiment_results = SentimentResultSerializer(many=True, read_only=True)

    class Meta:
        model = Product
        fields = ["id", "name", "created_at", "sentiment_results"]

