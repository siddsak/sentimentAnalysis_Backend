# sentiment_api/models.py
from django.db import models

class Product(models.Model):
    name = models.CharField(max_length=200)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

class SentimentResult(models.Model):
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='sentiment_results')
    total_reviews = models.IntegerField()
    positive_reviews = models.IntegerField()
    negative_reviews = models.IntegerField()
    neutral_reviews = models.IntegerField()
    average_rating = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Sentiment Result for {self.product.name}"