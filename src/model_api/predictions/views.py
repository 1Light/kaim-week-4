import joblib
import os
import numpy as np
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import PredictionRequestSerializer
from django.conf import settings

# Load the trained model
MODEL_PATH = os.path.join(settings.BASE_DIR, '..', '..', 'scripts', 'sales_prediction_model_10-01-2025-22-31-15-013546.pkl')
model = joblib.load(MODEL_PATH)

class PredictSales(APIView):
    def post(self, request):
        serializer = PredictionRequestSerializer(data=request.data)
        if serializer.is_valid():
            # Extract data
            data = serializer.validated_data
            features = np.array([[  
                data['store_id'],
                data['advertising_spend'],
                int(data['holiday']),
                data['product_category_A'],
                data['product_category_B'],
            ]])

            # Predict
            prediction = model.predict(features)[0]

            return Response({'predicted_sales': prediction}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)