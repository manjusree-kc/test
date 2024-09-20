from django.contrib import admin
from django.urls import path
from reg_app.views import *

urlpatterns = [
    path("admin/", admin.site.urls),
    path("statistics-view/", statistics_view ),
    path("regulations-list/", regulations_list ),
    path("get_regulation_details/", get_regulation_details, name="get_regulation_details"),
    path("get_impact_level_pie_data/", get_impact_level_pie_data, name="get_impact_level_pie_data"),
    path("get_notifications_count_per_body/", get_notifications_count_per_body, name="get_notifications_count_per_body"),
    path("get_quarterly_data/", get_quarterly_data, name="get_quarterly_data"),



    
]
