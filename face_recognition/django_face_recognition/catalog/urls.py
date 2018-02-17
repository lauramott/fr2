from django.conf.urls import url
from . import views

urlpatterns = [
    url('', views.HomePage, name='home-page'),
    url('photo/', views.PhotoModel, name='photo'),
]
