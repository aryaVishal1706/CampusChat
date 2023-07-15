

from django.contrib import admin
from django.urls import path
from chatbot import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),
    path('chatbot/', views.chatbot, name='chatbot'),
]

# urlpatterns = [
#      path('', views.index, name='index'),
#     path('chatbot/', views.chatbot, name='chatbot'),
# ]

# from django.urls import path
# from . import views

# urlpatterns = [
#     path('', views.index, name='index'),
# ]


