from django.urls import path,include
from .views import face_swap,home_page

urlpatterns = [
    path('face_swap/',face_swap),
    path('',home_page, name="home_page"),
]