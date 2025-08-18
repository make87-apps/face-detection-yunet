import logging
import time
from importlib.resources import files

import cv2
import numpy as np
from make87_messages.detection.box.box_2d_pb2 import Box2DAxisAligned
from make87_messages.geometry.box.box_2d_aligned_pb2 import Box2DAxisAligned as Box2DAxisAlignedGeometry
from make87_messages.detection.box.boxes_2d_pb2 import Boxes2DAxisAligned
from make87_messages.image.uncompressed.any_pb2 import ImageRawAny
from make87_messages.core.header_pb2 import Header
from make87.encodings import ProtobufEncoder
from make87.interfaces.zenoh import ZenohInterface

logging.Formatter.converter = time.gmtime
logging.basicConfig(
    format="[%(asctime)sZ %(levelname)s  %(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%dT%H:%M:%S"
)


def convert_raw_image_to_bgr(message: ImageRawAny) -> np.ndarray:
    """
    Convert ImageRawAny message to BGR format for OpenCV processing.
    Supports RGB888, RGBA8888, and YUV420 formats.
    """
    # Check which image format is present in the oneof field
    format_field = message.WhichOneof('image')

    if format_field == 'rgb888':
        rgb_image = message.rgb888
        width, height = rgb_image.width, rgb_image.height
        # Convert RGB888 data to numpy array and reshape
        rgb_array = np.frombuffer(rgb_image.data, dtype=np.uint8).reshape((height, width, 3))
        # Convert RGB to BGR for OpenCV
        bgr_image = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        return bgr_image

    elif format_field == 'rgba8888':
        rgba_image = message.rgba8888
        width, height = rgba_image.width, rgba_image.height
        # Convert RGBA8888 data to numpy array and reshape
        rgba_array = np.frombuffer(rgba_image.data, dtype=np.uint8).reshape((height, width, 4))
        # Convert RGBA to BGR for OpenCV (dropping alpha channel)
        bgr_image = cv2.cvtColor(rgba_array, cv2.COLOR_RGBA2BGR)
        return bgr_image

    elif format_field == 'yuv420':
        yuv_image = message.yuv420
        width, height = yuv_image.width, yuv_image.height
        # YUV420 format: Y plane (width*height) + U plane (width*height/4) + V plane (width*height/4)
        yuv_data = np.frombuffer(yuv_image.data, dtype=np.uint8)

        # YUV420 planar format layout
        y_size = width * height
        uv_size = width * height // 4

        # Extract Y, U, V planes
        y_plane = yuv_data[:y_size].reshape((height, width))
        u_plane = yuv_data[y_size:y_size + uv_size].reshape((height // 2, width // 2))
        v_plane = yuv_data[y_size + uv_size:y_size + 2 * uv_size].reshape((height // 2, width // 2))

        # Upsample U and V planes to full resolution
        u_full = cv2.resize(u_plane, (width, height), interpolation=cv2.INTER_LINEAR)
        v_full = cv2.resize(v_plane, (width, height), interpolation=cv2.INTER_LINEAR)

        # Combine Y, U, V planes
        yuv_full = np.stack([y_plane, u_full, v_full], axis=2)

        # Convert YUV to BGR
        bgr_image = cv2.cvtColor(yuv_full.astype(np.uint8), cv2.COLOR_YUV2BGR)
        return bgr_image

    else:
        raise ValueError(f"Unsupported image format: {format_field}. Supported formats: rgb888, rgba8888, yuv420")


def main():
    # Initialize encoders for message types
    image_encoder = ProtobufEncoder(message_type=ImageRawAny)
    bbox_encoder = ProtobufEncoder(message_type=Boxes2DAxisAligned)

    # Initialize zenoh interface
    zenoh_interface = ZenohInterface(name="zenoh")

    # Get subscriber and publisher
    image_subscriber = zenoh_interface.get_subscriber("IMAGE_DATA")
    bbox_publisher = zenoh_interface.get_publisher("BOUNDING_BOXES_2D")

    model_path = files("app") / "res" / "face_detection_yunet_2023mar.onnx"
    face_detector = cv2.FaceDetectorYN.create(model=str(model_path), config="", input_size=(0, 0))
    
    if face_detector is None:
        raise RuntimeError(f"Failed to create face detector with model: {model_path}")

    # Cache variables
    previous_orig_size = None
    previous_input_size = None
    scale_matrix = None

    def process_message(message: ImageRawAny):
        nonlocal previous_orig_size, previous_input_size, scale_matrix

        # Convert the raw image message to BGR format for OpenCV processing
        image = convert_raw_image_to_bgr(message)
        format_field = message.WhichOneof('image')
        logging.debug(f"Processing image in {format_field} format")

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
        result, faces = face_detector.detect(resized)
        
        if result != 1:
            logging.warning(f"Face detection failed with result code: {result}")
            return

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
                # Ensure face detection output has enough elements
                if len(face) < 15:
                    logging.warning(f"Face detection output has {len(face)} elements, expected 15")
                    continue

                # Create header for individual bounding box
                bbox_header = Header()
                bbox_header.CopyFrom(header)
                bbox_header.entity_path = f"{header.entity_path}/{i}"

                bbox_geom_header = Header()
                bbox_geom_header.CopyFrom(bbox_header)
                bbox_geom_header.entity_path = f"{bbox_header.entity_path}/geometry"

                bbox_2d = Box2DAxisAligned(
                    header=bbox_header,
                    geometry=Box2DAxisAlignedGeometry(
                        x=face[0],
                        y=face[1],
                        width=face[2],
                        height=face[3],
                        header=bbox_geom_header,
                    ),
                    confidence=face[14],
                    class_id=0,
                )
                bboxes_2d.boxes.append(bbox_2d)

            # Encode and publish the message
            encoded_message = bbox_encoder.encode(bboxes_2d)
            bbox_publisher.put(payload=encoded_message)
            
            # Safe timestamp logging
            timestamp_str = "unknown"
            try:
                if header.timestamp:
                    timestamp_str = header.timestamp.ToDatetime()
            except Exception as e:
                logging.debug(f"Error formatting timestamp: {e}")
                
            logging.info(f"Published {len(bboxes_2d.boxes)} bounding boxes for frame with timestamp {timestamp_str}")

    # Subscribe to incoming messages
    for sample in image_subscriber:
        try:
            if sample.payload is None:
                logging.warning("Received sample with None payload")
                continue
                
            payload_bytes = sample.payload.to_bytes()
            if not payload_bytes:
                logging.warning("Received empty payload")
                continue
                
            message = image_encoder.decode(payload_bytes)
            process_message(message)
        except Exception as e:
            logging.error(f"Error processing message: {e}")


if __name__ == "__main__":
    main()
