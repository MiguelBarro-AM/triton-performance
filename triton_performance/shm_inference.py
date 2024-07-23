import time

import numpy as np
import tritonclient.http as httpclient
import tritonclient.utils.shared_memory as shm
import ctypes
import cv2
import triton_performance.common as common
from multiprocessing import Process, Queue, cpu_count

url = 'localhost:8000'
client = httpclient.InferenceServerClient(url=url, verbose=False)


class Model:
    def __init__(self, model_name, model_version, input_name, output_name):
        self.model_name = model_name
        self.model_version = model_version
        self.input_name = input_name
        self.output_name = output_name

    def __str__(self):
        return f"Model name: {self.model_name}, Model version: {self.model_version}"


class SharedMemoryRegion:
    def __init__(self, name, key, size):
        self.name = name
        self.key = key
        self.size = size
        self.handle = None

    def initialize_shared_memory_region(self):
        self.handle = shm.create_shared_memory_region(self.name, self.key, self.size)
        client.register_system_shared_memory(self.name, self.key, self.size)

        return self.handle

    def __str__(self):
        return f"Name: {self.name}, Key: {self.key}, Size: {self.size}"


def initialize_shared_memory_region(region_name: str, region_key: str, region_size: int,):
    shm_handle = shm.create_shared_memory_region(region_name, region_key, region_size)
    # add "-v /dev/shm:/dev/shm" to docker run (tritonserver), if not the following will fail
    client.register_system_shared_memory(region_name, region_key, region_size)

    return shm_handle


def read_image(img_path):
    img_bytes = cv2.imread(img_path)
    img = common.image_conversion(img_bytes, 512, 384)

    return img


def dispose_shared_memory_region(shm_region: SharedMemoryRegion):
    client.unregister_system_shared_memory(shm_region.name)
    shm.destroy_shared_memory_region(shm_region.handle)


def send_request(queue, img, batch_size: int, input_shm: SharedMemoryRegion, output_shm: SharedMemoryRegion, model: Model):
    while True:
        start_time = time.time()
        inputs = []
        outputs = []
        images = []

        for _ in range(batch_size):
            images.append(img)

        for i, img in enumerate(images):
            shm.set_shared_memory_region(input_shm.handle, [img], offset=i * int(input_shm.size/batch_size))

        inputs.append(httpclient.InferInput(model.input_name, [batch_size, 384, 512, 3], "FP32"))
        inputs[0].set_shared_memory(input_shm.name, input_shm.size)

        outputs.append(httpclient.InferRequestedOutput(model.output_name))
        outputs[0].set_shared_memory(output_shm.name, output_shm.size)

        _ = client.infer(model_name=model.model_name, inputs=inputs, outputs=outputs)

        outputs = shm.get_contents_as_numpy(output_shm.handle, np.float32, [batch_size, 1, 2])

        queue.put((start_time, time.time() - start_time))


def main():
    queue = Queue()
    num_processes = 1
    batch_size = 8
    processes = []

    input_shm_name = 'input_shm'
    output_shm_name = 'output_shm'
    input_shm_key = '/input_shm_region'
    output_shm_key = '/output_shm_region'

    client.unregister_system_shared_memory(input_shm_name)
    client.unregister_system_shared_memory(output_shm_name)

    input_shm = SharedMemoryRegion(input_shm_name,
                                   input_shm_key,
                                   batch_size * 1 * 3 * 512 * 384 * ctypes.sizeof(ctypes.c_float))

    output_shm = SharedMemoryRegion(output_shm_name,
                                    output_shm_key,
                                    batch_size * 1 * 2 * ctypes.sizeof(ctypes.c_float))

    input_shm.initialize_shared_memory_region()
    output_shm.initialize_shared_memory_region()

    model = Model('emissions_poland', '1', 'input_1', 'Vector_clasificador_final')

    img = read_image("triton_performance/resources/image.jpg")

    for _ in range(num_processes):
        p = Process(target=send_request, args=(queue, img, batch_size, input_shm, output_shm, model, ))
        p.start()
        processes.append(p)

    logger = Process(target=common.log_results, args=(queue, batch_size, ))
    logger.start()

    for p in processes:
        p.join()
        dispose_shared_memory_region(input_shm)
        dispose_shared_memory_region(output_shm)

    logger.join()


if __name__ == '__main__':
    main()
