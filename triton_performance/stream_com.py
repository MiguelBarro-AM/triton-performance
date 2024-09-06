import asyncio
import time
import tritonclient.grpc.aio as tc

from triton_performance.common import InferenceModel


async def async_request_iterator(model, max_sequence, batch_size=1):
    for sequence_id in range(max_sequence):
        input_image = model.image

        inputs = [tc.InferInput(
            model.input_name,
            [batch_size, model.dims[1], model.dims[2], model.dims[3]], model.dtype)]
        inputs[0].set_data_from_numpy(input_image)

        outputs = [tc.InferRequestedOutput(model.output_name)]

        yield {
            "model_name": model.name,
            "model_version": model.version,
            "inputs": inputs,
            "outputs": outputs,
            "request_id": str(sequence_id),
            "sequence_id": sequence_id,
            "sequence_start": sequence_id == 0,
            "sequence_end": sequence_id == batch_size - 1
        }


async def run(queue, server_url: str, image: str, model: InferenceModel, batch_size: int = 1):
    async with tc.InferenceServerClient(url=server_url, verbose=False) as client:
        _ = await model.get_metadata(client)
        _ = model.get_image(image, batch_size)

        while True:
            start_time = time.time()
            request_iterator = async_request_iterator(model, 10, batch_size)

            try:
                response_iterator = client.stream_infer(
                    inputs_iterator=request_iterator,
                    stream_timeout=600000,
                )

                if not hasattr(response_iterator, '__aiter__'):
                    raise TypeError("response_iterator is not an asynchronous iterable.")

                async for response in response_iterator:
                    try:
                        result, error = response
                        if error:
                            raise error

                        _ = result.as_numpy(model.output_name)
                        queue.put((start_time, time.time() - start_time))

                    except Exception as e:
                        print(f"Error processing response: {e}")
                        queue.put(None)

            except asyncio.CancelledError:
                print(f"CancelledError")
            except Exception as e:
                print(f"Exception: {e}")
