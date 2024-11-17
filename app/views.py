from django.http import JsonResponse
from .tasks import run



def index(request):
     run.delay()
     return JsonResponse({"message": "Hello, world. You're at the polls index."})