from django.urls import path
from . import views

urlpatterns = [
    path('all_cards/', views.all_cards_view, name='all_cards'),
    path('alzheimer/', views.alzheimer_view, name='alzheimer'),
    path('brain/', views.brain_view, name='brain'),
    path('diabetes/', views.diabetes_view, name='diabetes'),
    path('general/', views.general_view, name='general'),
    path('heart/', views.heart_view, name='heart'),
    path('kidney/', views.kidney_view, name='kidney'),
    path('predict/', views.general_view, name='predict_disease'),
    path('speak/', views.speak, name='speak'),
]
