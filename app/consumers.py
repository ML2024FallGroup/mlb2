import json
from channels.generic.websocket import WebsocketConsumer
from asgiref.sync import async_to_sync
from .tasks import run

class Consumer(WebsocketConsumer):
    def connect(self):
        self.room_group_name = 'test'

        async_to_sync(self.channel_layer.group_add)(
            self.room_group_name,
            self.channel_name
        )

        self.accept()
        async_to_sync(self.channel_layer.group_send)(
            self.room_group_name,
            {
                'type':'chat_message',
                'message': 'loud and clear'
            }
        )
        # run.delay()
        
    def chat_message(self, event):
        message = event['message']

        self.send(text_data=json.dumps({
            'type':'chat',
            'message':message
        }))
    
    def main_message(self, event):
        detail = event['detail']
        _type = event['item_type']

        self.send(text_data=json.dumps({
            'type': 'chat',
            'item_type': _type,
            'detail': detail
        }))

    def stem_message(self, event):
        detail = event['detail']
        _type = event['item_type']

        self.send(text_data=json.dumps({
            'type': 'chat',
            'item_type': _type,
            'detail': detail
        }))