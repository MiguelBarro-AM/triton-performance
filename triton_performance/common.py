import cv2
import numpy as np


def image_conversion(img, width, height):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    img = np.asarray(img, dtype='float32')
    img /= 255
    img = img.flatten()

    return img


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
