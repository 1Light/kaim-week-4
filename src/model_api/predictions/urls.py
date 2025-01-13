from django.urls import path
from .views import PredictSales 

urlpatterns = [
    path('predict/', PredictSales.as_view(), name='predict-sales'),
]