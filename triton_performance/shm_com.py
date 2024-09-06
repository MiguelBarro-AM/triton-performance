import time

import numpy as np
import tritonclient.grpc.aio as tc
import tritonclient.utils.shared_memory as shm
import ctypes

from triton_performance.common import InferenceModel


class SharedMemoryRegion:
    def __init__(self, name, key, size):
        self.name = name
        self.key = key
        self.size = size
        self.handle = None

    async def initialize_shared_memory_region(self, client: tc.InferenceServerClient):
        self.handle = shm.create_shared_memory_region(self.name, self.key, self.size)
        await client.register_system_shared_memory(self.name, self.key, self.size)

        return self.handle

    def __str__(self):
        return f"Name: {self.name}, Key: {self.key}, Size: {self.size}"


def dispose_shared_memory_region(client: tc.InferenceServerClient, shm_region: SharedMemoryRegion):
    client.unregister_system_shared_memory(shm_region.name)
    shm.destroy_shared_memory_region(shm_region.handle)


async def run(queue, server_url: str, image: str, model: InferenceModel, batch_size: int = 1):
    async with tc.InferenceServerClient(url=server_url, verbose=False) as client:
        _ = await model.get_metadata(client)
        _ = model.get_image(image, batch_size)

        input_shm_name = 'input_shm'
        output_shm_name = 'output_shm'
        input_shm_key = '/input_shm_region'
        output_shm_key = '/output_shm_region'

        await client.unregister_system_shared_memory(input_shm_name)
        await client.unregister_system_shared_memory(output_shm_name)

        input_shm_size = batch_size * model.dims[1] * model.dims[2] * model.dims[3] * ctypes.sizeof(ctypes.c_float)
        output_shm_size = batch_size * model.class_count * ctypes.sizeof(ctypes.c_float)

        input_shm = SharedMemoryRegion(input_shm_name, input_shm_key, input_shm_size)
        output_shm = SharedMemoryRegion(output_shm_name, output_shm_key, output_shm_size)

        await input_shm.initialize_shared_memory_region(client)
        await output_shm.initialize_shared_memory_region(client)

        while True:
            start_time = time.time()
            inputs = []
            outputs = []

            shm.set_shared_memory_region(input_shm.handle, [model.image], offset=0)

            inputs.append(tc.InferInput(model.input_name, model.dims, model.dtype))
            inputs[0].set_shared_memory(input_shm.name, input_shm.size)

            outputs.append(tc.InferRequestedOutput(model.output_name))
            outputs[0].set_shared_memory(output_shm.name, output_shm.size)

            try:
                response = await client.infer(model_name=model.name, inputs=inputs, outputs=outputs)
            except tc.InferenceServerException as e:
                print(f"Inference error: {e}")
                continue

            try:
                outputs_np = shm.get_contents_as_numpy(output_shm.handle, np.float32, [batch_size, model.class_count])
            except Exception as e:
                print(f"Error reading output: {e}")

            queue.put((start_time, time.time() - start_time))
