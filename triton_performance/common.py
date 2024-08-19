import cv2
import numpy as np


def log_results(queue, batch_size):
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
                print(f"Requests per second: {(count/span)*batch_size:.4f} / Average time: {avg_time/count:.4f}")
                count = 0
                avg_time = 0

        if duration is not None:
            avg_time += duration[1]
            count += 1
        if duration is None:
            print("Failed request")


def get_model(model_name):
    if model_name == "emissions_poland":
        return Model(model_name, "1", "input_1", "Vector_clasificador_final")
    elif model_name == "global_classifier":
        return Model(model_name, "1", "input.1", "1463")
    else:
        raise ValueError("Model not found")


def get_image(model_name, image_path):
    if model_name == "emissions_poland":
        img = Image(512, 384, 3)
        img.read_image(image_path)

        return img

    elif model_name == "global_classifier":
        img = Image(224, 224, 3)
        img.read_image(image_path)

        return img


class Image:
    def __init__(self, width, height, channels):
        self.width = width
        self.height = height
        self.channels = channels
        self.image = None

    def read_image(self, image_path):
        self.image = cv2.imread(image_path)
        return self.image

    def image_conversion(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)
        img = np.asarray(img, dtype='float32')
        img /= 255
        img = img.flatten()

        return img

    def get_shape(self):
        return [1, self.height, self.width, self.channels]


class Model:
    def __init__(self, name, version, class_count):
        self.name = name
        self.version = version
        self.class_count = class_count

    def __str__(self):
        return f"Model name: {self.name}, Model version: {self.version}"
