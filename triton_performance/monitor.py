import requests
import time
import click

TRITON_METRICS_URL = 'http://localhost:8002/metrics'


def get_inference_count(model_name: str, version: str):
    response = requests.get(TRITON_METRICS_URL)
    if response.status_code == 200:
        results = []
        metrics = response.text
        for line in metrics.split('\n'):
            if 'nv_inference_request_success{{model="{}",version="{}"}}'.format(model_name, version) in line:
                results.append(int(line.split()[-1]))
            elif 'nv_gpu_utilization{gpu_uuid' in line:
                results.append(float(line.split()[-1])*100)

        return results

    return None


@click.command()
@click.option("-m", "--model_name", help="Model name")
@click.option("-v", "--version", default="1", help="Model version")
def monitor_inferences(model_name, version):
    results = get_inference_count(model_name, version)
    prev_count = results[0]
    prev_time = time.time()

    if prev_count is None:
        print("Unable to retrieve metrics.")
        return

    while True:
        time.sleep(5)
        results = get_inference_count(model_name, version)

        if results is not None:
            current_count = results[0]
            current_time = time.time()

            diff_time = current_time - prev_time
            infer_per_second = (current_count - prev_count)/diff_time
            print(f"Requests per second: {infer_per_second:.4f} / Total GPU use %: {results[1]:.0f}")
            prev_count = current_count
            prev_time = current_time
        else:
            print("Unable to retrieve metrics.")


if __name__ == "__main__":
    monitor_inferences()
