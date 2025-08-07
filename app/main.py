import logging
import time
from importlib.resources import files

import cv2
import numpy as np
from make87_messages.geometry.box.box_2d_aligned_pb2 import Box2DAxisAligned
from make87_messages.geometry.box.boxes_2d_aligned_pb2 import Boxes2DAxisAligned
from make87_messages.image.compressed.image_jpeg_pb2 import ImageJPEG
from make87_messages.core.header_pb2 import Header
from make87.encodings import ProtobufEncoder
from make87.interfaces.zenoh import ZenohInterface

logging.Formatter.converter = time.gmtime
logging.basicConfig(
    format="[%(asctime)sZ %(levelname)s  %(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%dT%H:%M:%S"
)


def main():
    # Initialize encoders for message types
    image_encoder = ProtobufEncoder(message_type=ImageJPEG)
    bbox_encoder = ProtobufEncoder(message_type=Boxes2DAxisAligned)

    # Initialize zenoh interface
    zenoh_interface = ZenohInterface(name="zenoh")

    # Get subscriber and publisher
    image_subscriber = zenoh_interface.get_subscriber("IMAGE_DATA")
    bbox_publisher = zenoh_interface.get_publisher("BOUNDING_BOXES_2D")

    model_path = files("app") / "res" / "face_detection_yunet_2023mar.onnx"
    face_detector = cv2.FaceDetectorYN.create(model=str(model_path), config="", input_size=(0, 0))

    # Cache variables
    previous_orig_size = None
    previous_input_size = None
    scale_matrix = None

    def process_message(message: ImageJPEG):
        nonlocal previous_orig_size, previous_input_size, scale_matrix

        image = cv2.imdecode(np.frombuffer(message.data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        orig_height, orig_width, _ = image.shape

        # Check if we need to update anything
        if previous_orig_size != (orig_width, orig_height):
            target_width = 960
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
            # Create header based on original message header
            header = Header()
            if message.header:
                header.CopyFrom(message.header)
                # Append entity path for faces
                if header.entity_path:
                    header.entity_path = header.entity_path + "/faces"
                else:
                    header.entity_path = "/faces"
            else:
                header.entity_path = "/faces"
                header.reference_id = 0

            # Don't update timestamp to preserve original timing

            faces[:, :4] *= scale_matrix  # Fast vectorized scaling

            bboxes_2d = Boxes2DAxisAligned(header=header, boxes=[])
            for i, face in enumerate(faces):
                # Create header for individual bounding box
                bbox_header = Header()
                bbox_header.CopyFrom(header)
                bbox_header.entity_path = f"{header.entity_path}/{i}"

                bbox_2d = Box2DAxisAligned(
                    x=face[0],
                    y=face[1],
                    width=face[2],
                    height=face[3],
                    header=bbox_header,
                )
                bboxes_2d.boxes.append(bbox_2d)

            # Encode and publish the message
            encoded_message = bbox_encoder.encode(bboxes_2d)
            bbox_publisher.put(payload=encoded_message)
            logging.info(f"Published {len(bboxes_2d.boxes)} bounding boxes for frame with timestamp {header.timestamp.ToDatetime()}")

    # Subscribe to incoming messages
    for sample in image_subscriber:
        try:
            message = image_encoder.decode(sample.payload.to_bytes())
            process_message(message)
        except Exception as e:
            logging.error(f"Error processing message: {e}")


if __name__ == "__main__":
    main()
