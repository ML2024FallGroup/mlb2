from django.http import JsonResponse
from uuid import uuid4
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.core.files.storage import default_storage
from pathlib import Path
from django.conf import settings
import os

from .tasks import process_audio



def index(request):
     return JsonResponse({"message": "Hello, world. You're at the polls index."})

class AudioProcessingView(APIView):
     ALLOWED_AUDIO_TYPES = {
        'audio/mpeg': '.mp3',
        'audio/wav': '.wav',
        'audio/x-wav': '.wav',
        'audio/ogg': '.ogg',
        'audio/x-m4a': '.m4a',
        'audio/aac': '.aac',
        'audio/mp3': '.mp3',  # Some clients might send this
        'audio/mpeg3': '.mp3',  # Some clients might send this
        'video/mp4': '.mp4',  # Some clients might send audio as mp4
        'application/octet-stream': '.mp3'  # Generic binary data
    }
     
     def validate_audio_file(self, file):
          # Check if file exists
          if not file:
               return False, "No file provided"
               
          # Check file type
          content_type = file.content_type
          if content_type not in self.ALLOWED_AUDIO_TYPES:
               return False, f"Invalid file type. Allowed types: {', '.join(self.ALLOWED_AUDIO_TYPES.keys())}"
               
          # Check file size (e.g., 50MB limit)
          if file.size > 50 * 1024 * 1024:  # 50MB in bytes
               return False, "File too large. Maximum size is 50MB"
               
          return True, None

     def save_audio_file(self, file):
          # Create a unique filename
          ext = self.ALLOWED_AUDIO_TYPES.get(file.content_type, '.wav')
          filename = f"{uuid4()}{ext}"
          
          # Create uploads directory if it doesn't exist
          upload_dir = Path(settings.MEDIA_ROOT) / 'audio_uploads'
          upload_dir.mkdir(parents=True, exist_ok=True)
          
          # Save the file
          file_path = upload_dir / filename
          with default_storage.open(str(file_path), 'wb+') as destination:
               for chunk in file.chunks():
                    destination.write(chunk)
                    
          return str(file_path)
     
     def cleanup_file(self, file_path):
          try:
               os.remove(file_path)
          except Exception as e:
               print(f"Error cleaning up file {file_path}: {e}")


     def post(self, request):
          audio_file = request.FILES.get('audio')
          
          # Validate the file
          is_valid, error_message = self.validate_audio_file(audio_file)
          if not is_valid:
               return Response(
                    {'error': error_message},
                    status=status.HTTP_400_BAD_REQUEST
               )
          
          try:
               # Save the file temporarily
               file_path = self.save_audio_file(audio_file)

               process_audio.delay(file_path)
               
               return Response({
                    'status': 'Processing started',
                    'location': str(file_path)
               })
          

               
          except Exception as e:
               return Response(
                    {'error': f'Error processing audio: {str(e)}'},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
               )
