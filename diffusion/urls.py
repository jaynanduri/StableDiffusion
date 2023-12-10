from django.urls import re_path
from .views import *

urlpatterns = [
    re_path(r'generate$', post_generated_image, name='generate'),
]