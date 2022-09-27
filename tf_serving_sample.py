import tensorflow as tf
import numpy as np
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from aiogrpc import insecure_channel


class RequestRestApi(object):
    def __init__(self, host_name: str, model_name: str):
        self.endpoint = host_name
        self.model_name = model_name

    async def __aenter__(self):
        self._channel = insecure_channel(self.endpoint)
        self._stub = prediction_service_pb2_grpc.PredictionServiceStub(self._channel)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self._channel.close()

    async def predict(self, image: np.ndarray):
        self.request = predict_pb2.PredictRequest()
        self.request.model_spec.name = self.model_name
        # 'input_1' is your graph model's first layer name
        self.request.inputs['input_1'].CopyFrom(
        tf.make_tensor_proto(image, shape=[1, 224, 224, 3])) 
        return await self._stub.Predict(self.request, 5.0)


async def async_function_sample(image: np.ndarray):
    async with RequestRestApi('0.0.0.0:8500', 'test_model2') as session:
	    output = await session.predict(image=image)
        # 'output' is your graph model's last layer name
        # output = output.outputs['output'].float_val
        # output = np.array(output)
        # output = np.reshape(output, (224, 224, 3))