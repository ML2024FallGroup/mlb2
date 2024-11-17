from celery import shared_task
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
import time

@shared_task
def run():
  channel_layer = get_channel_layer()
  for _ in range(11):
    async_to_sync(channel_layer.group_send)(
            'test',
            {
                'type': 'chat_message',
                'message': 'proceed'
            }
        )
    time.sleep(3)
    