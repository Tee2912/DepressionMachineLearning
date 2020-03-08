# api/serializers.py

from rest_framework import serializers
from .models import mlmodel

class mlmodelSerializer(serializers.ModelSerializer):
    """Serializer to map the Model instance into JSON format."""

    class Meta:
        """Meta class to map serializer's fields with the model fields."""
        model = mlmodel
        fields = '__all__'