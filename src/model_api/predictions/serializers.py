from rest_framework import serializers

class PredictionRequestSerializer(serializers.Serializer):
    store_id = serializers.IntegerField()
    advertising_spend = serializers.FloatField()
    holiday = serializers.BooleanField()
    product_category_A = serializers.IntegerField()
    product_category_B = serializers.IntegerField()