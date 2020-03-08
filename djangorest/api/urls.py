from django.conf.urls import url, include
from rest_framework.urlpatterns import format_suffix_patterns
from api import views

urlpatterns = [
    url(r'^mlmodel/', views.mlmodel.as_view()),
]

# urlpatterns = format_suffix_patterns(urlpatterns)