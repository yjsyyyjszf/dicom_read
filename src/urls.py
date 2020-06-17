from django.contrib.auth.decorators import login_required
from django.urls import path

from src.views import HomeView, LoginView, LogoutView, ExportView, PrintView

urlpatterns = [
    path('', LoginView.as_view(), name='login'),
    path('logout', LogoutView.as_view(), name='logout'),
    path('home/', login_required(HomeView.as_view()), name='home'),
    path('export/', login_required(ExportView.as_view()), name='export'),
    path('print/', login_required(PrintView.as_view()), name='print')
]
