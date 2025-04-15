# search/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='search_home'),  # this is the default page of the app
    path('chatbot/', views.chatbot, name='chatbot'),
]
