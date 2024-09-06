import cv2
import numpy as np
import tritonclient.grpc.aio as tc
import tritonclient.grpc.model_config_pb2 as mc


def log_results(queue, batch_size):
    start_time = 0
    count = 0
    avg_time = 0
    while True:
        duration = queue.get()
        # Show results
        span = duration[0] - start_time
        if span > 5:
            start_time = duration[0]
            if count != 0:
                print(f"Requests per second: {(count/span)*batch_size:.4f} / Average time: {avg_time/count:.4f}")
                count = 0
                avg_time = 0

        if duration is not None:
            avg_time += duration[1]
            count += 1
        if duration is None:
            print("Failed request")


class InferenceModel:
    def __init__(self, name, version, class_count):
        self.name = name
        self.version = version
        self.class_count = class_count
        self.model_metadata = None
        self.model_config = None
        self.input_format = None
        self.dtype = None
        self.input_name = None
        self.output_name = None
        self.dims = None
        self.image = None

    def __str__(self):
        return f"Model name: {self.name}, Model version: {self.version}"

    async def get_metadata(self, client: tc.InferenceServerClient):
        model_metadata = await client.get_model_metadata(self.name, self.version)
        model_config = await client.get_model_config(self.name, self.version)
        model_config = model_config.config

        # checking inputs
        input_metadata = model_metadata.inputs[0]
        input_name = input_metadata.name
        input_config = model_config.input[0]

        # Model input must have 3 dims, either CHW or HWC (not counting
        # the batch dimension), either CHW or HWC
        max_channels = 4
        max_batch_size = model_config.max_batch_size
        expected_input_dims = len(input_metadata.shape)
        batch_dim = expected_input_dims == max_channels

        if input_config.format == mc.ModelInput.FORMAT_NONE:
            shape = input_metadata.shape[1:] if batch_dim else input_metadata.shape
            if shape[0] == min(shape):
                input_config.format = mc.ModelInput.FORMAT_NCHW
            else:
                input_config.format = mc.ModelInput.FORMAT_NHWC

        input_format = input_config.format

        if input_format == mc.ModelInput.FORMAT_NHWC:
            height = input_metadata.shape[1 if batch_dim else 0]
            width = input_metadata.shape[2 if batch_dim else 1]
            channels = input_metadata.shape[3 if batch_dim else 2]
        else:
            channels = input_metadata.shape[1 if batch_dim else 0]
            height = input_metadata.shape[2 if batch_dim else 1]
            width = input_metadata.shape[3 if batch_dim else 2]

        # checking outputs
        output_metadata = model_metadata.outputs[0]

        output_name = output_metadata.name

        self.model_metadata = model_metadata
        self.model_config = model_config
        self.input_format = input_format
        self.dtype = input_metadata.datatype
        self.input_name = input_name
        self.output_name = output_name
        self.dims = [max_batch_size, channels, height, width]

    def get_image(self, image_path, batch_size=1):
        img = cv2.imread(image_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.dims[3], self.dims[2]), interpolation=cv2.INTER_AREA)
        img = np.asarray(img, dtype='float32')
        img = img.reshape(self.dims[1], self.dims[2], self.dims[3])
        img /= 255.0

        img = np.expand_dims(img, axis=0)
        img = np.repeat(img, batch_size, axis=0)

        self.image = img
        return img
