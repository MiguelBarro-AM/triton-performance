import grpc
import requests
import time
import cv2
import io
import numpy as np
from PIL import Image
from multiprocessing import Process, Queue, cpu_count

from tritonclient.grpc import service_pb2, service_pb2_grpc
from tritonclient.utils import triton_to_np_dtype
import tritonclient.grpc.model_config_pb2 as mc
from pathlib import Path


def image_conversion(img, width, height):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    img = np.asarray(img, dtype='float32')
    img /= 255
    img = img.flatten()

    return img


def common_request(img, model_name, model_version):
    img = image_conversion(img, 512, 384)

    # Generate the request
    request = service_pb2.ModelInferRequest()
    request.model_name = model_name
    request.model_version = model_version

    # Populate the inputs in inference request
    input0 = service_pb2.ModelInferRequest().InferInputTensor()
    input0.name = "input_1"
    input0.datatype = "FP32"
    input0.shape.extend([1, 384, 512, 3])
    for i in img:
        input0.contents.fp32_contents.append(i)

    request.inputs.extend([input0])

    return request


def send_request(queue, img, options):
    channel = grpc.insecure_channel("localhost:8001")
    grpc_stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)

    request = common_request(img, "emissions_poland", "1")

    while True:
        start_time = time.time()
        response = grpc_stub.ModelInfer(request)
        # grpc_stub.ModelInfer(request)

        if response is not None:
             queue.put((start_time, time.time() - start_time))
        else:
             queue.put(None)


def log_results(queue):
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
                print(f"Requests per second: {count/span:.4f} / Average time: {avg_time/count:.4f}")
                count = 0
                avg_time = 0

        if duration is not None:
            avg_time += duration[1]
            count += 1
        if duration is None:
            print("Failed request")


def main():
    queue = Queue()
    # num_processes = cpu_count() - 1
    num_processes = 10
    processes = []

    img_bytes = cv2.imread("triton_performance/resources/image.jpg")
    options = [
        ('grpc.max_send_message_length', 20 * 1024 * 1024),
        ('grpc.max_receive_message_length', 1024),
        ('grpc.max_concurrent_streams', 100000),
        ('grpc.keepalive_time_ms', 10000),
        ('grpc.keepalive_timeout_ms', 5000),
        ('grpc.http2.min_time_between_pings_ms', 10000),
        ('grpc.http2.max_pings_without_data', 0),
    ]

    for _ in range(num_processes):
        p = Process(target=send_request, args=(queue, img_bytes, options, ))
        p.start()
        processes.append(p)

    logger = Process(target=log_results, args=(queue,))
    logger.start()

    for p in processes:
        p.join()

    logger.join()


if __name__ == "__main__":
    main()
