import grpc
import cv2
import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.grpc import InferInput, InferRequestedOutput, InferenceServerException
from triton_performance.common import Image


url = "localhost:8001"
model_name = "trucks_onnx"
label_file = '/home/pablo/models/trucks_onnx/trucks_labels.txt'

timeout = 10
input_width = 640
input_height = 640


def load_labels(label_file):
    labels = []
    with open(label_file, 'r') as f:
        for line in f:
            labels.append(line.strip())
    return labels


def main():
    try:
        confidence = 0
        max_confidence = 0
        predicted_label = "Unknown"
        x, y, w, h = 0, 0, 0, 0

        labels = load_labels(label_file)
        triton_client = grpcclient.InferenceServerClient(url=url)

        if not triton_client.is_server_ready():
            raise Exception("Triton server not ready!")
        if not triton_client.is_model_ready(model_name):
            raise Exception(f"Model {model_name} not ready!")

        image = Image(640, 640, 3)
        image.read_image("/home/pablo/Imágenes/trucks/output.png")
        original_height, original_width = image.image.shape[:2]
        input_data = image.image_conversion()

        print("Forma original de la imagen:", input_data.shape)

        input_data = input_data.reshape(1, 3, 640, 640)

        input_name = "images"
        inputs = [InferInput(input_name, input_data.shape, "FP32")]
        inputs[0].set_data_from_numpy(input_data)

        output_name = "output0"
        outputs = [InferRequestedOutput(output_name)]

        response = triton_client.infer(model_name, inputs=inputs, outputs=outputs, timeout=timeout)

        output_data = response.as_numpy(output_name)
        print("Forma de output_data:", output_data.shape)

        batch_size, num_boxes, num_values = output_data.shape

        assert num_values == 10, "Expected 10 values for each box, obtained {}.".format(num_values)

        for i in range(num_boxes):
            box = output_data[0, i, :]

            confidence = box[4]
            class_probs = box[5:]

            if confidence > max_confidence:
                max_confidence = confidence
                predicted_class_index = np.argmax(class_probs)
                predicted_label = labels[predicted_class_index]

                x, y, w, h = box[:4]
                x = int(x * original_width / input_width)
                y = int(y * original_height / input_height)
                w = int(w * original_width / input_width)
                h = int(h * original_height / input_height)

        print(f"Predicted label: {predicted_label}, Confidence: {max_confidence}, Box: {x}, {y}, {w}, {h}")
        cv2.rectangle(image.image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        label = f"{predicted_label}: {max_confidence:.4f}"
        cv2.putText(image.image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow('Detections', image.image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except InferenceServerException as e:
        print(f"Triton Server Exception: {e}")
    except grpc.RpcError as e:
        print(f"Error in gRPC request: {e}")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
