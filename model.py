import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.builders import model_builder
from object_detection.utils import config_util
import cv2
import numpy as np
import base64
import os
import threading
import requests

configs = config_util.get_configs_from_pipeline_file(
    os.path.join("models/new", "pipeline.config")
)
detection_model = model_builder.build(model_config=configs["model"], is_training=True)
checkpoint = tf.compat.v2.train.Checkpoint(model=detection_model)
checkpoint.restore(os.path.join("models", "new", "ckpt-3")).expect_partial()

category_index = label_map_util.create_categories_from_labelmap(
    os.path.join("models", "annotations", "label_map.pbtxt")
)


@tf.function
def detect_function(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


def detect_object(image_path):
    img = cv2.imread(image_path)
    image_np = np.array(img)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

    detections = detect_function(input_tensor)

    num_detections = int(detections.pop("num_detections"))
    detections = {
        key: value[0, :num_detections].numpy() for key, value in detections.items()
    }
    detections["num_detections"] = num_detections

    detections["detection_classes"] = detections["detection_classes"].astype(np.int64)
    # for i in range(num_detections):
    class_id = int(detections["detection_classes"][0])
    score = float(detections["detection_scores"][0])
    image_width, image_height = img.shape[1], img.shape[0]
    ymin, xmin, ymax, xmax = detections["detection_boxes"][0]
    cv2.rectangle(
        img,
        (int(xmin * image_width), int(ymin * image_height)),
        (int(xmax * image_width), int(ymax * image_height)),
        (0, 255, 0),
        2,
    )
    _, buffer = cv2.imencode(".jpg", img)
    image_base64 = base64.b64encode(buffer).decode("utf-8")
    return {
        "name": category_index[class_id]["name"],
        "score": score,
        "image": image_base64,
    }


def send_feedback(host, data):
    resp = requests.post(f"{host}/feedback", data=data)


def live_detect_object(img, host):
    image_np = np.array(img)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_function(input_tensor)
    num_detections = int(detections.pop("num_detections"))
    detections = {
        key: value[0, :num_detections].numpy() for key, value in detections.items()
    }
    detections["num_detections"] = num_detections
    detections["detection_classes"] = detections["detection_classes"].astype(np.int64)
    for i in range(len(detections["detection_scores"])):
        class_id = int(detections["detection_classes"][i])
        score = float(detections["detection_scores"][i])
        if score > 0.6:
            feedback_thread = threading.Thread(
                target=send_feedback, args=[host, category_index[class_id]["name"]]
            )
            feedback_thread.start()
            image_height, image_width, _ = img.shape
            ymin, xmin, ymax, xmax = detections["detection_boxes"][i]
            cv2.rectangle(
                img,
                (int(xmin * image_width), int(ymin * image_height)),
                (int(xmax * image_width), int(ymax * image_height)),
                (0, 255, 0),
                2,
            )
            cv2.putText(
                img,
                category_index[class_id]["name"] + str(score),
                (int(xmin * image_width), int(ymin * image_height) + 20),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 255, 0),
            )

    return img


if __name__ == "__main__":
    image = r"C:\Users\IMPANO\Desktop\zegot\workspace\training_demo\images\train\bean_rust.jpg"
    print(detect_object(image))


# import tensorflow as tf
# import cv2
# import numpy as np
# import base64
# import os
# import threading
# import requests
# from object_detection.utils import label_map_util


# class DetectionModelWrapper(tf.keras.Model):
#     def __init__(self, detection_model, **kwargs):
#         super(DetectionModelWrapper, self).__init__(**kwargs)
#         self.detection_model = detection_model

#     def call(self, inputs):
#         image, shapes = self.detection_model.preprocess(inputs)
#         prediction_dict = self.detection_model.predict(image, shapes)
#         detections = self.detection_model.postprocess(prediction_dict, shapes)
#         return detections

#     @classmethod
#     def from_config(cls, config):
#         detection_model = tf.keras.models.model_from_config(config["detection_model"])
#         return cls(detection_model)

#     def get_config(self):
#         config = super(DetectionModelWrapper, self).get_config()
#         config["detection_model"] = self.detection_model.get_config()
#         return config


# # Load the saved .h5 model with custom_objects
# detection_model = tf.keras.models.load_model(
#     "detection_model.h5",
#     custom_objects={"DetectionModelWrapper": DetectionModelWrapper},
# )

# # Load the label map
# category_index = label_map_util.create_categories_from_labelmap(
#     os.path.join("models", "annotations", "label_map.pbtxt")
# )


# # Define the detect function using the loaded .h5 model
# def detect_function(image):
#     predictions = detection_model(image)  # Use the loaded Keras model
#     return predictions


# def detect_object(image_path):
#     img = cv2.imread(image_path)
#     image_np = np.array(img)

#     # Preprocess image (expand dimensions and normalize)
#     input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

#     # Run detection
#     detections = detect_function(input_tensor)

#     # Postprocess detections
#     num_detections = int(detections["num_detections"])
#     detections = {
#         key: value[0, :num_detections].numpy() for key, value in detections.items()
#     }
#     detections["num_detections"] = num_detections
#     detections["detection_classes"] = detections["detection_classes"].astype(np.int64)

#     # Extract detection results
#     class_id = int(detections["detection_classes"][0])
#     score = float(detections["detection_scores"][0])
#     image_width, image_height = img.shape[1], img.shape[0]
#     ymin, xmin, ymax, xmax = detections["detection_boxes"][0]

#     # Draw bounding box on the image
#     cv2.rectangle(
#         img,
#         (int(xmin * image_width), int(ymin * image_height)),
#         (int(xmax * image_width), int(ymax * image_height)),
#         (0, 255, 0),
#         2,
#     )

#     # Encode image to base64
#     _, buffer = cv2.imencode(".jpg", img)
#     image_base64 = base64.b64encode(buffer).decode("utf-8")

#     # Return results
#     return {
#         "name": category_index[class_id]["name"],
#         "score": score,
#         "image": image_base64,
#     }


# def send_feedback(host, data):
#     resp = requests.post(f"{host}/feedback", data=data)


# def live_detect_object(img, host):
#     image_np = np.array(img)

#     # Preprocess image (expand dimensions and normalize)
#     input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

#     # Run detection
#     detections = detect_function(input_tensor)

#     # Postprocess detections
#     num_detections = int(detections.pop("num_detections"))
#     detections = {
#         key: value[0, :num_detections].numpy() for key, value in detections.items()
#     }
#     detections["num_detections"] = num_detections
#     detections["detection_classes"] = detections["detection_classes"].astype(np.int64)

#     # Loop through detections and process feedback
#     for i in range(len(detections["detection_scores"])):
#         class_id = int(detections["detection_classes"][i])
#         score = float(detections["detection_scores"][i])
#         if score > 0.6:
#             feedback_thread = threading.Thread(
#                 target=send_feedback, args=[host, category_index[class_id]["name"]]
#             )
#             feedback_thread.start()
#             image_height, image_width, _ = img.shape
#             ymin, xmin, ymax, xmax = detections["detection_boxes"][i]
#             cv2.rectangle(
#                 img,
#                 (int(xmin * image_width), int(ymin * image_height)),
#                 (int(xmax * image_width), int(ymax * image_height)),
#                 (0, 255, 0),
#                 2,
#             )
#             cv2.putText(
#                 img,
#                 category_index[class_id]["name"] + str(score),
#                 (int(xmin * image_width), int(ymin * image_height) + 20),
#                 cv2.FONT_HERSHEY_COMPLEX,
#                 1,
#                 (0, 255, 0),
#             )

#     return img


# if __name__ == "__main__":
#     image = r"C:\Users\IMPANO\Desktop\zegot\workspace\training_demo\images\train\bean_rust.jpg"
#     print(detect_object(image))
