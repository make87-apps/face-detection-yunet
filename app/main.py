import logging
import os

import cv2
import numpy as np
from make87_messages.image.compressed.image_jpeg_pb2 import ImageJPEG
from make87_messages.geometry.box.box_2d_pb2 import Box2DAxisAligned
from make87 import initialize, get_subscriber_topic, get_publisher_topic, resolve_topic_name


def main():
    initialize()
    image_topic_name = resolve_topic_name(name="IMAGE_DATA")
    bbox_2d_topic_name = resolve_topic_name(name="BOUNDING_BOX_2D")
    image_topic = get_subscriber_topic(name=image_topic_name, message_type=ImageJPEG)
    bbox_2d_topic = get_publisher_topic(name=bbox_2d_topic_name, message_type=Box2DAxisAligned)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "face_detection_yunet_2023mar.onnx")
    face_detector = cv2.FaceDetectorYN.create(model=model_path, config="", input_size=(0, 0))

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
