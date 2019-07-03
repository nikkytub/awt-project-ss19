from channels.generic.websocket import AsyncWebsocketConsumer
import json
import numpy as np
import tensorflow as tf


avg_length = 325

# To pad shorter and truncate longer conversation
vocabulary_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(avg_length)


def rnn_model_retrieve(mes):
    num_conversation = np.array(list(vocabulary_processor.fit_transform([mes])))
    sess = tf.Session()
    signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    input_key = 'x_input'
    output_key = 'y_output'
    export_path = '../model_rnn'
    meta_graph_def = tf.saved_model.loader.load(
        sess,
        ["myTag"],
        export_path)
    signature = meta_graph_def.signature_def
    x_tensor_name = signature[signature_key].inputs[input_key].name
    y_tensor_name = signature[signature_key].outputs[output_key].name
    x = sess.graph.get_tensor_by_name(x_tensor_name)
    y = sess.graph.get_tensor_by_name(y_tensor_name)
    y_out = sess.run(y, {x: num_conversation}).tolist()
    return y_out


class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.room_name = self.scope['url_route']['kwargs']['room_name']
        self.room_group_name = 'chat_%s' % self.room_name

        # Join room group
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )

        await self.accept()

    async def disconnect(self, close_code):
        # Leave room group
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )

    # Receive message from WebSocket
    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        message = text_data_json['message']

        # Send message to room group
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'chat_message',
                'message': message
            }
        )

    # Receive message from room group
    async def chat_message(self, event):
        message = event['message']
        data = rnn_model_retrieve(message)

        # Send message to WebSocket
        await self.send(text_data=json.dumps({
            'message': [message, data]
        }))
