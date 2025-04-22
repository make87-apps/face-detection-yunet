import logging
from importlib.resources import files

import cv2
import numpy as np
import make87
from make87_messages.geometry.box.box_2d_aligned_pb2 import Box2DAxisAligned
from make87_messages.geometry.box.boxes_2d_aligned_pb2 import Boxes2DAxisAligned
from make87_messages.image.compressed.image_jpeg_pb2 import ImageJPEG
from make87_messages.core.header_pb2 import Header


def main():
    make87.initialize()
    image_topic = make87.get_subscriber(name="IMAGE_DATA", message_type=ImageJPEG)
    bbox_2d_topic = make87.get_publisher(name="BOUNDING_BOXES_2D", message_type=Boxes2DAxisAligned)

    model_path = files("app") / "res" / "face_detection_yunet_2023mar.onnx"
    face_detector = cv2.FaceDetectorYN.create(model=str(model_path), config="", input_size=(0, 0))

    # Cache variables
    previous_orig_size = None
    previous_input_size = None
    scale_matrix = None

    def callback(message: ImageJPEG):
        nonlocal previous_orig_size, previous_input_size, scale_matrix

        image = cv2.imdecode(np.frombuffer(message.data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        orig_height, orig_width, _ = image.shape

        # Check if we need to update anything
        if previous_orig_size != (orig_width, orig_height):
            target_width = 640
            target_height = int((target_width / orig_width) * orig_height)
            previous_input_size = (target_width, target_height)
            previous_orig_size = (orig_width, orig_height)

            face_detector.setInputSize(previous_input_size)
            scale_matrix = np.array(
                [
                    orig_width / target_width,
                    orig_height / target_height,
                    orig_width / target_width,
                    orig_height / target_height,
                ]
            )
            logging.debug(f"Updated detector input size: {previous_input_size}")
            logging.debug(f"Updated scale matrix: {scale_matrix}")

        resized = cv2.resize(image, previous_input_size)
        _, faces = face_detector.detect(resized)

        if faces is not None and len(faces) > 0:
            header = make87.header_from_message(
                header_cls=Header,
                message=message,
                append_entity_path="faces",
                set_current_time=True,
            )

            faces[:, :4] *= scale_matrix  # Fast vectorized scaling

            bboxes_2d = Boxes2DAxisAligned(header=header, boxes=[])
            for i, face in enumerate(faces):
                bbox_2d = Box2DAxisAligned(
                    x=face[0],
                    y=face[1],
                    width=face[2],
                    height=face[3],
                    header=make87.header_from_message(
                        header_cls=Header,
                        message=bboxes_2d,
                        append_entity_path=f"{i}",
                        set_current_time=False,
                    ),
                )
                bboxes_2d.boxes.append(bbox_2d)

            bbox_2d_topic.publish(bboxes_2d)
            logging.info(f"Published {len(bboxes_2d.boxes)} bounding boxes")

    image_topic.subscribe(callback)
    make87.loop()


if __name__ == "__main__":
    main()
