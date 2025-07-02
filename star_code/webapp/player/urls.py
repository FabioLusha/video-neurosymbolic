from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.video_upload, name='video_upload'),
    path('', views.video_list, name='video_list'),
] 