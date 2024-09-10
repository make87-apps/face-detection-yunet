import os

import cv2
import numpy as np
from make87_messages.image.ImageJPEG_pb2 import ImageJPEG
from make87_messages.geometry.BoundingBox2D_pb2 import AxisAlignedBoundingBox2DFloat
from make87 import get_topic, topic_names, PublisherTopic


def main():
    image_topic = get_topic(name=topic_names.IMAGE_DATA)
    bbox_2d_topic = get_topic(name=topic_names.BOUNDING_BOX_2D)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "face_detection_yunet_2023mar.onnx")
    face_detector = cv2.FaceDetectorYN.create(model=model_path, config="", input_size=(0, 0))

    def callback(message: ImageJPEG):
        image = cv2.imdecode(np.frombuffer(message.data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        print(f"Received image with shape: {image.shape}")

        height, width, _ = image.shape
        face_detector.setInputSize((width, height))

        _, faces = face_detector.detect(image)
        faces = faces if faces is not None else []
        for face in faces:
            bbox_2d = AxisAlignedBoundingBox2DFloat(
                timestamp=message.timestamp, x=face[0], y=face[1], width=face[2], height=face[3]
            )
            bbox_2d_topic.publish(bbox_2d)
            print(f"Published bounding box: {bbox_2d}")

    image_topic.subscribe(callback)


if __name__ == "__main__":
    main()
