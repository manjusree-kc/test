from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.decorators import api_view
from .models import RegulationsList, RegulationsDetails
from .serializers import RegulationsListSerializer
from django.http import JsonResponse

from django.http import JsonResponse
from .models import *
from django.core.serializers import serialize
from django.db.models import Count
from django.db.models import Count, Q
from datetime import datetime, timedelta
import json

def regulations_list(request):
    try:
        queryset = RegulationsList.objects.all()
        data = serialize('json', queryset)
        
        data = json.loads(data)
        
        formatted_data = []
        for item in data:
            obj = item['fields']
            obj['not_id'] = item['pk']  # Include primary key if needed
            formatted_data.append(obj)

        # Return as JSON response
        return JsonResponse({'status': True, 'message': 'success', 'data': formatted_data})
    except:
        return JsonResponse({'status': False, 'message': 'failed'}, status=404)


def get_regulation_details(request, not_id):
    try:
        # Query the model for the record with the specified not_id
        obj = RegulationsDetails.objects.get(not_id=not_id)
        
        data = {
            "not_id": obj.not_id,
            "chapter_no": obj.chapter_no,
            "notification_doc_url": obj.notification_doc_url,
            "notification_doc_with_highlight_url": obj.notification_doc_with_highlight_url,
            "highlighted_pages": obj.highlighted_pages,
            "notification_summary": obj.notification_summary,
            "insights": obj.insights
        }

        return JsonResponse({'status': True, 'message': 'success', 'data': data}, status=200)
    except RegulationsDetails.DoesNotExist:
        return JsonResponse({'status': False, 'message': 'failed'}, status=404)
    

def get_impact_level_pie_data(request):
    try:
        impact_counts = RegulationsList.objects.values('impact_label').annotate(count=Count('impact_label'))
        total_count = RegulationsList.objects.count()

        pie_data = []

        for impact in impact_counts:
            impact_label = impact['impact_label']
            count = impact['count']
            percentage = (count / total_count) * 100 if total_count > 0 else 0
            pie_data.append({
                'impact_label': impact_label,
                'percentage': round(percentage, 2)
            })
            print(pie_data)

        return JsonResponse({'status': True, 'message': 'success', 'title': 'Impact Level','data': pie_data})
    except:
        return JsonResponse({'status': False, 'message': 'failed'}, status=404)


def get_notifications_count_per_body(request):
    try:
        body_counts = RegulationsList.objects.values('regulatory_body').annotate(count=Count('regulatory_body'))
        bar_data = []

        for body in body_counts:
            regulatory_body = body['regulatory_body']
            count = body['count']
            bar_data.append({
                'regulatory_body': regulatory_body,
                'count': count
            })

        return JsonResponse({'status': True, 'message': 'success', 'title': 'Notifications Per Regulatory Body','data': bar_data})
    except:
        return JsonResponse({'status': False, 'message': 'failed'}, status=404)

def get_quarterly_data(request):
    try:
        now = datetime.now()
        # get the start and end of a given quarter
        def get_quarter_dates(year, quarter):
            start_month = (quarter - 1) * 3 + 1
            start_date = datetime(year, start_month, 1)
            end_date = start_date + timedelta(days=90) - timedelta(days=1)
            return start_date, end_date

        # current year and quarter
        current_quarter = (now.month - 1) // 3 + 1
        current_year = now.year

        # the start and end of the last four quarters including the current one
        quarters = [(current_year, current_quarter)]
        for i in range(1, 4):
            prev_quarter = (current_quarter - i - 1) % 4 + 1
            prev_year = current_year - ((current_quarter - i - 1) // 4)
            quarters.append((prev_year, prev_quarter))

        data = []

        for year, quarter in reversed(quarters):
            start_date, end_date = get_quarter_dates(year, quarter)
            total_count = RegulationsList.objects.filter(timestamp__range=[start_date, end_date]).count()
            action_taken_count = RegulationsList.objects.filter(
                timestamp__range=[start_date, end_date],
                action_status=True
            ).count()
            no_action_count = RegulationsList.objects.filter(
                timestamp__range=[start_date, end_date],
                action_status=False
            ).count()

            data.append({
                'quarter': f'Q{quarter} {year}',
                'total': total_count,
                'action_taken': action_taken_count,
                'no_action': no_action_count
            })

        return JsonResponse({'status': True, 'message': 'success', 'title': 'Quarterly Notification Data','data': data})
    except:
        return JsonResponse({'status': False, 'message': 'failed'}, status=404)



@api_view(['GET'])
def statistics_view(request):
    try:
        total_regulators = RegulationsList.objects.values('regulatory_body').distinct().count()
        total_notifications = RegulationsList.objects.count()
        high_impact_count = RegulationsList.objects.filter(impact_label='high').count()
        action_required = RegulationsList.objects.filter(action_status=False).count()

        data = {
            "total_regulators": total_regulators,
            "total_notifications": total_notifications,
            "high_impact_count": high_impact_count,
            "action_required": action_required,
        }
        return JsonResponse({'status': True, 'message': 'success', 'title': 'Quarterly Notification Data','data': data})
    
    except:
        return JsonResponse({'status': False, 'message': 'failed'}, status=404)

  




