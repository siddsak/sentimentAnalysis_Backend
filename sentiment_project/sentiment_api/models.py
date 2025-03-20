# # sentiment_api/models.py
# from django.db import models

# class Product(models.Model):
#     name = models.CharField(max_length=200)
#     created_at = models.DateTimeField(auto_now_add=True)

#     def __str__(self):
#         return self.name

# # class SentimentResult(models.Model):
# #     product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='sentiment_results')
# #     total_reviews = models.IntegerField()
# #     positive_reviews = models.IntegerField()
# #     negative_reviews = models.IntegerField()
# #     neutral_reviews = models.IntegerField()
# #     average_rating = models.FloatField()
# #     created_at = models.DateTimeField(auto_now_add=True)

# #updated
# class SentimentResult(models.Model):
#     product = models.ForeignKey(Product, related_name='sentiment_results', on_delete=models.CASCADE)
#     total_reviews = models.IntegerField()
#     average_rating = models.FloatField()
#     positive_reviews = models.IntegerField()
#     negative_reviews = models.IntegerField()
#     neutral_reviews = models.IntegerField()
#     model_comparison = models.JSONField(null=True, blank=True)
#     created_at = models.DateTimeField(auto_now_add=True)

#     def __str__(self):
#         return f"Sentiment Result for {self.product.name}"

# sentiment_api/models.py
from django.db import models
import json

class Product(models.Model):
    name = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.name

class SentimentResult(models.Model):
    product = models.ForeignKey(
        Product, 
        related_name='sentiment_results', 
        on_delete=models.CASCADE
    )
    total_reviews = models.IntegerField(default=0)
    positive_reviews = models.IntegerField(default=0)
    negative_reviews = models.IntegerField(default=0)
    neutral_reviews = models.IntegerField(default=0)
    average_rating = models.FloatField(default=0.0)
    model_comparison = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Sentiment Analysis for {self.product.name} - {self.created_at}"
    
    def set_model_comparison(self, data):
        self.model_comparison = json.dumps(data)
    
    def get_model_comparison(self):
        if self.model_comparison:
            return json.loads(self.model_comparison)
        return {}