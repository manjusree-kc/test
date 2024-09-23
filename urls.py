from django.urls import path
from .views import *

urlpatterns = [
    path('get_regulation/<int:not_id>/', get_regulation_details, name='get_regulation_details'),
    path('get_table_data/', get_table_data, name='get_table_data'),
    path('statistics_view/', statistics_view, name='statistics_view'),
    path("get_impact_level_pie_data/", get_impact_level_pie_data, name="get_impact_level_pie_data"),
    path("get_notifications_count_per_body/", get_notifications_count_per_body, name="get_notifications_count_per_body"),
    path("get_quarterly_data/", get_quarterly_data, name="get_quarterly_data"),
    path("calculate_response_time_intervals/", calculate_response_time_intervals, name="calculate_response_time_intervals")
]
