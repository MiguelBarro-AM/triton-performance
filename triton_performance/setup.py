import click

from triton_performance.common import get_model, get_image, log_results
from triton_performance.grpc_com import send_request

from multiprocessing import Process, Queue


def start_process(queue, image, model, communication):
    if communication == "grpc":
        send_request(queue, image, model)


@click.command()
@click.option("-c", "--communication", default="grpc", help="Communication protocol")
@click.option("-p", "--processes", default=1, help="Number of processes to run")
@click.option("-b", "--batch_size", default=8, help="Batch size")
@click.option("-m", "--model_name", help="Model name")
@click.option("-i", "--image", help="Image path")
def main(num_processes, batch_size, model_name, image_path, communication):
    queue = Queue()
    processes = []

    model = get_model(model_name)
    image = get_image(model_name, image_path)

    for i in range(num_processes):
        p = Process(target=start_process, args=(queue, image, model, communication))
        p.start()
        processes.append(p)

    logger = Process(target=log_results, args=(queue, batch_size,))
    logger.start()

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
