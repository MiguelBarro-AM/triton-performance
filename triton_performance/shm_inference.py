import numpy as np
import tritonclient.http as httpclient
import tritonclient.utils.shared_memory as shm
import ctypes
import cv2
import triton_performance.common as common

url = 'localhost:8000'
client = httpclient.InferenceServerClient(url=url, verbose=True)


def initialize_shared_memory_region(region_name: str, region_key: str, region_size: int,):
    shm_handle = shm.create_shared_memory_region(region_name, region_key, region_size)
    # add "-v /dev/shm:/dev/shm" to docker run (tritonserver), if not the following will fail
    client.register_system_shared_memory(region_name, region_key, region_size)

    return shm_handle


def main():

    input_shm_name = 'input_shm'
    output_shm_name = 'output_shm'
    input_shm_key = '/input_shm_region'
    output_shm_key = '/output_shm_region'

    client.unregister_system_shared_memory(input_shm_name)
    client.unregister_system_shared_memory(output_shm_name)

    input_size = 1 * 3 * 512 * 384 * ctypes.sizeof(ctypes.c_float)
    input_shm_handle = initialize_shared_memory_region(input_shm_name, input_shm_key, input_size)

    output_size = 1 * 2 * ctypes.sizeof(ctypes.c_float)
    output_shm_handle = initialize_shared_memory_region(output_shm_name, output_shm_key, output_size)

    img_bytes = cv2.imread("./resources/image.jpg")
    img = common.image_conversion(img_bytes, 512, 384)

    shm.set_shared_memory_region(input_shm_handle, [img])

    inputs = []
    outputs = []

    inputs.append(httpclient.InferInput('input_1', [1, 384, 512, 3], "FP32"))
    inputs[0].set_shared_memory(input_shm_name, input_size)

    outputs.append(httpclient.InferRequestedOutput('Vector_clasificador_final'))
    outputs[0].set_shared_memory(output_shm_name, output_size)

    results = client.infer(model_name='emissions_poland', inputs=inputs, outputs=outputs)

    output_data = shm.get_contents_as_numpy(output_shm_handle, np.float32, [1, 2])
    print(output_data)

    client.unregister_system_shared_memory(input_shm_name)
    client.unregister_system_shared_memory(output_shm_name)

    shm.destroy_shared_memory_region(input_shm_handle)
    shm.destroy_shared_memory_region(output_shm_handle)


if __name__ == '__main__':
    main()
