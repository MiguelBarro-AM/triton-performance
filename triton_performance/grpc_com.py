import asyncio
import time
import tritonclient.grpc.aio as tc

from triton_performance.common import InferenceModel


async def run(queue, server_url: str, image: str, model: InferenceModel, batch_size: int = 1):
    async with tc.InferenceServerClient(url=server_url, verbose=False) as client:
        _ = await model.get_metadata(client)
        _ = model.get_image(image, batch_size)

        while True:
            start_time = time.time()
            inputs = [tc.InferInput(
                model.input_name,
                [batch_size, model.dims[1], model.dims[2], model.dims[3]], model.dtype)]
            inputs[0].set_data_from_numpy(model.image)

            outputs = [tc.InferRequestedOutput(model.output_name)]

            try:
                response = await client.infer(model_name=model.name, inputs=inputs, outputs=outputs)

                if response is not None:
                    _ = response.as_numpy(model.output_name)
                    queue.put((start_time, time.time() - start_time))
                else:
                    queue.put(None)

            except asyncio.CancelledError:
                print(f"CancelledError")
            except Exception as e:
                print(f"Exception: {e}")
