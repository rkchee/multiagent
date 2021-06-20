from transformers import ElectraTokenizer, TFElectraModel

from ray import serve
import requests
import ray


class ElectraTensors:
    def __init__(self):
        from transformers import ElectraTokenizer, TFElectraModel
        self.tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator', max_length=128, pad_to_max_length=True)
        self.model = TFElectraModel.from_pretrained('google/electra-small-discriminator')

    def __call__(self, flask_request):
        import tensorflow as tf
        import json
        self.answer = flask_request.json
        self.input_ids = tf.constant(self.tokenizer.encode(self.answer['text'], max_length=128, pad_to_max_length=128))[None, :]  # Batch size 1
        outputs = self.model(self.input_ids)
        results = outputs[0].numpy().tolist()
        return {
            "Electra_Tensors": results
        }


#connect to ray start --head
# url="34.83.237.208"
# url="35.233.235.19"
url = "35.230.63.77"
ray.init(address= url + ":6000", _redis_password='5241590000000000')
client = serve.start(http_host="0.0.0.0")


client.create_backend('electra_tensors', ElectraTensors)
client.create_endpoint('ElectraTensors', backend='electra_tensors', route='/electratensors', methods=['POST'])
