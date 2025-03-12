import logging
from importlib.resources import files

import cv2
import numpy as np
import make87 as m87
from make87_messages.geometry.box.box_2d_aligned_pb2 import Box2DAxisAligned
from make87_messages.image.compressed.image_jpeg_pb2 import ImageJPEG
from make87_messages.core.header_pb2 import Header

def main():
    m87.initialize()
    image_topic = m87.get_subscriber(name="IMAGE_DATA", message_type=ImageJPEG)
    bbox_2d_topic = m87.get_publisher(name="BOUNDING_BOX_2D", message_type=Box2DAxisAligned)

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
            header = m87.header_from_message(header_cls=Header, message=message, append_entity_path="face", set_current_time=True)
            bbox_2d = Box2DAxisAligned(x=face[0], y=face[1], width=face[2], height=face[3], header=header)
            bbox_2d_topic.publish(bbox_2d)
            logging.info(f"Published bounding box: {bbox_2d}")

    image_topic.subscribe(callback)
    m87.loop()


if __name__ == "__main__":
    main()
