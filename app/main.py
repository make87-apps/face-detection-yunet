import logging
from importlib.resources import files

import cv2
import numpy as np
from make87 import initialize, get_subscriber, get_publisher, resolve_topic_name
from make87_messages.geometry.box.box_2d_pb2 import Box2DAxisAligned
from make87_messages.image.compressed.image_jpeg_pb2 import ImageJPEG


def main():
    initialize()
    image_topic = get_subscriber(name="IMAGE_DATA", message_type=ImageJPEG)
    bbox_2d_topic = get_publisher(name="BOUNDING_BOX_2D", message_type=Box2DAxisAligned)

    model_path = files("app") / "res" / "face_detection_yunet_2023mar.onnx"
    face_detector = cv2.FaceDetectorYN.create(model=str(model_path), config="", input_size=(0, 0))

    def callback(message: ImageJPEG):
        image = cv2.imdecode(np.frombuffer(message.data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        logging.info(f"Received image with shape: {image.shape}")

        height, width, _ = image.shape
        face_detector.setInputSize((width, height))

        _, faces = face_detector.detect(image)
        faces = faces if faces is not None else []
        for face in faces:
            bbox_2d = Box2DAxisAligned(timestamp=message.timestamp, x=face[0], y=face[1], width=face[2], height=face[3])
            bbox_2d_topic.publish(bbox_2d)
            logging.info(f"Published bounding box: {bbox_2d}")

    image_topic.subscribe(callback)


if __name__ == "__main__":
    main()
