import asyncio
import time

import tritonclient.grpc.aio as tc
import tritonclient.grpc.model_config_pb2 as mc

from triton_performance.common import Image, Model


async def get_model_data(client: tc.InferenceServerClient,
                         model_name: str,
                         version: str,
                         class_count: int) -> dict:

    model_metadata = await client.get_model_metadata(model_name, version)
    model_config = await client.get_model_config(model_name, version)
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

    return {
        "name": model_name,
        "version": version,
        "class_count": class_count,
        "metadata": model_metadata,
        "config": model_config,
        "input_format": input_format,
        "dtype": input_metadata.datatype,
        "input_name": input_name,
        "output_name": output_name,
        "dims": (channels, height, width, max_batch_size),
    }


async def async_message_iterator(img):
    request_id = 0

    while True:
        await asyncio.sleep(0.001)
        request_id += 1

        yield {
            "request_id": request_id,
            "input_image": img
        }


async def async_inference_iterator(metadata, image, input_iterator: object):

    shape = image.get_shape()

    async for input_dict in input_iterator:
        inference_input = tc.InferInput(metadata["input_name"], shape, "FP32")
        inference_input.set_data_from_numpy(input_dict["input_image"])

        inference_output = tc.InferRequestedOutput(metadata["output_name"], metadata["class_count"])

        yield {
            "model_name": metadata["name"],
            "model_version": metadata["version"],
            "inputs": [inference_input],
            "outputs": [inference_output],
            "request_id": str(input_dict["request_id"])
        }


async def send_request(queue, image: Image, model: Model):
    start_time = time.time()

    client = tc.InferenceServerClient("localhost:8001")
    metadata = await get_model_data(client, model.name, model.version, model.class_count)

    img = image.image_conversion()
    request_iterator = async_message_iterator(img)
    response_iterator = async_inference_iterator(metadata, image, request_iterator)

    async for response in response_iterator:
        result, error = response

        if error:
            raise error

    queue.put((start_time, time.time() - start_time))
