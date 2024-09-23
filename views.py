from django.http import JsonResponse
from .models import NotificationData
from django.core.serializers import serialize
from django.db.models import Count, Q
from datetime import datetime, timedelta
from django.utils import timezone
from rest_framework.decorators import api_view


@api_view(['GET'])
def statistics_view(request):
    try:
        total_regulators = NotificationData.objects.values('regulatory_body').distinct().count()
        total_notifications = NotificationData.objects.count()
        high_impact_count = NotificationData.objects.filter(impact_label='High').count()
        action_required = NotificationData.objects.filter(action_status=False).count()

        data = {
            "total_regulators": total_regulators,
            "total_notifications": total_notifications,
            "high_impact_count": high_impact_count,
            "action_required": action_required,
        }
        return JsonResponse({'status': True, 'message': 'success', 'title': 'Quarterly Notification Data','data': data})
    
    except:
        return JsonResponse({'status': False, 'message': 'failed'}, status=404)

@api_view(['GET'])
def get_table_data(request):
    try:
        required_columns = ["not_id", "subpart", "regulatory_body",
                            "timestamp", "chapter_no", "part_name",
                            "impact_label", "impact_value", "action_status"]
        queryset = NotificationData.objects.values(*required_columns)
        data = list(queryset)

        return JsonResponse({'status': True, 'message': 'success', 'data': data})
    except Exception as e:
        return JsonResponse({'status': False, 'message': str(e)}, status=500)

@api_view(['GET'])
def get_regulation_details(request, not_id):
    try:
        data_obj = NotificationData.objects.get(not_id=not_id)
        
        data = {
            "not_id": not_id,
            "chapter_name": data_obj.chapter_name,
            "part_name": data_obj.part_name,
            "subpart": data_obj.subpart,
            "regulatory_body": data_obj.regulatory_body,
            "timestamp": data_obj.timestamp,
            "impact_label": data_obj.impact_label,
            "impact_value": data_obj.impact_value,
            "action_status": data_obj.action_status,
            "action_taken_timestamp": data_obj.action_taken_timestamp,
            "notification_doc_url": data_obj.notification_doc_url,
            "notification_doc_with_highlight_url": data_obj.notification_doc_with_highlight_url,
            "highlighted_pages": data_obj.highlighted_pages,
            "notification_summary": data_obj.notification_summary,
            "insights": data_obj.insights
        }

        return JsonResponse({'status': True, 'message': 'success', 'data': data}, status=200)
    except NotificationData.DoesNotExist:
        return JsonResponse({'status': False, 'message': 'Regulation not found'}, status=404)
    except Exception as e:
        return JsonResponse({'status': False, 'message': str(e)}, status=500)

@api_view(['GET'])
def get_impact_level_pie_data(request):
    try:
        impact_counts = NotificationData.objects.values('impact_label').annotate(count=Count('impact_label'))
        total_count = NotificationData.objects.count()

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
    except Exception as e:
        return JsonResponse({'status': False, 'message': str(e)}, status=500)

@api_view(['GET'])
def get_notifications_count_per_body(request):
    try:
        body_counts = NotificationData.objects.values('regulatory_body').annotate(count=Count('regulatory_body'))
        bar_data = []

        for body in body_counts:
            regulatory_body = body['regulatory_body']
            count = body['count']
            bar_data.append({
                'regulatory_body': regulatory_body,
                'count': count
            })

        return JsonResponse({'status': True, 'message': 'success', 'title': 'Notifications Per Regulatory Body','data': bar_data})
    except Exception as e:
        return JsonResponse({'status': False, 'message': str(e)}, status=500)

@api_view(['GET'])
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
            total_count = NotificationData.objects.filter(timestamp__range=[start_date, end_date]).count()
            action_taken_count = NotificationData.objects.filter(
                timestamp__range=[start_date, end_date],
                action_status=True
            ).count()
            no_action_count = NotificationData.objects.filter(
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
    except Exception as e:
        return JsonResponse({'status': False, 'message': str(e)}, status=500)

@api_view(['GET'])
def calculate_response_time_intervals(request):
    try:
        now = timezone.now()

        intervals = {
            "0-1 days": 0,
            "1-2 days": 0,
            "2-5 days": 0,
            "5-10 days": 0,
            "10+ days": 0
        }

        # Fetch all notifications
        notifications = NotificationData.objects.all()

        for notification in notifications:
            if notification.action_taken_timestamp:
                response_time = (notification.action_taken_timestamp - notification.timestamp).days
                
                # Categorize response times into intervals
                if response_time <= 1:
                    intervals["0-1 days"] += 1
                elif response_time <= 2:
                    intervals["1-2 days"] += 1
                elif response_time <= 5:
                    intervals["2-5 days"] += 1
                elif response_time <= 10:
                    intervals["5-10 days"] += 1
                else:
                    intervals["10+ days"] += 1

        return JsonResponse({'status': True, 'message': 'success', 'title': 'Response Time Summary', 'data': intervals})
    
    except Exception as e:
        return JsonResponse({'status': False, 'message': str(e)}, status=500)
