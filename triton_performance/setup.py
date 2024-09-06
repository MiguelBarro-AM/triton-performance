import click
import asyncio

import triton_performance.stream_com as stream_com
import triton_performance.shm_com as shm_com
import triton_performance.grpc_com as grpc_com

from triton_performance.common import InferenceModel, log_results

from multiprocessing import Process, Queue


def start_process(queue, image, model, communication, batch_size, server):
    if communication == "stream":
        asyncio.run(stream_com.run(queue, server, image, model, batch_size))
    elif communication == "grpc":
        asyncio.run(grpc_com.run(queue, server, image, model, batch_size))
    elif communication == "http":
        raise NotImplementedError("HTTP communication not implemented")
    elif communication == "shm":
        asyncio.run(shm_com.run(queue, server, image, model, batch_size))
    else:
        raise ValueError("Invalid communication protocol")


def load_model(model_name):
    if model_name == "emissions_poland":
        return InferenceModel("emissions_poland", "1", 2)
    elif model_name == "global_classifier":
        return InferenceModel("global_classifier_dynamic", "1", 25)
    else:
        raise ValueError("Model not found")


@click.command()
@click.option("-c", "--communication", default="grpc", help="Communication protocol")
@click.option("-p", "--num_processes", default=2, help="Number of processes to run")
@click.option("-b", "--batch_size", default=8, help="Batch size")
@click.option("-m", "--model_name", help="Model name")
@click.option("-i", "--image_path", help="Image path")
@click.option("-s", "--server", default="localhost:8001", help="Server address '<host>:<port>'")
def main(communication, num_processes, batch_size, model_name, image_path, server):
    queue = Queue()
    processes = []

    model = load_model(model_name)

    for i in range(num_processes):
        p = Process(target=start_process, args=(queue, image_path, model, communication, batch_size, server))
        processes.append(p)
        p.start()

    logger = Process(target=log_results, args=(queue, batch_size,))
    logger.start()

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
