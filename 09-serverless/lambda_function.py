import json
import numpy as np
import onnxruntime as ort
from keras_image_helper import create_preprocessor

def preprocess_pytorch(X):
    X = X / 255.0

    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

    X = X.transpose(0, 3, 1, 2)
    X = (X - mean) / std

    return X.astype(np.float32)


preprocessor = create_preprocessor(preprocess_pytorch, target_size=(200, 200))

session = ort.InferenceSession("hair_classifier_empty.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


def lambda_handler(event, context):
    url = event["url"]
    X = preprocessor.from_url(url)

    y_pred = session.run([output_name], {input_name: X})[0]
    score = float(y_pred[0][0])

    return {"score": score}
