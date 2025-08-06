import logging
import time
from importlib.resources import files

import cv2
import numpy as np
from make87_messages.geometry.box.box_2d_aligned_pb2 import Box2DAxisAligned
from make87_messages.geometry.box.boxes_2d_aligned_pb2 import Boxes2DAxisAligned
from make87_messages.image.uncompressed.any_pb2 import ImageRawAny
from make87_messages.core.header_pb2 import Header
from make87.encodings import ProtobufEncoder
from make87.interfaces.zenoh import ZenohInterface

logging.Formatter.converter = time.gmtime
logging.basicConfig(
    format="[%(asctime)sZ %(levelname)s  %(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%dT%H:%M:%S"
)


def convert_raw_image_to_opencv(message: ImageRawAny) -> np.ndarray:
    """Convert ImageRawAny message to OpenCV BGR format."""

    if message.HasField("rgb888"):
        # RGB888 format
        rgb_data = message.rgb888
        height, width = rgb_data.height, rgb_data.width

        # Convert bytes to numpy array and reshape
        image_array = np.frombuffer(rgb_data.data, dtype=np.uint8)
        image = image_array.reshape((height, width, 3))

        # Convert RGB to BGR for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    elif message.HasField("rgba8888"):
        # RGBA8888 format
        rgba_data = message.rgba8888
        height, width = rgba_data.height, rgba_data.width

        # Convert bytes to numpy array and reshape
        image_array = np.frombuffer(rgba_data.data, dtype=np.uint8)
        image = image_array.reshape((height, width, 4))

        # Convert RGBA to BGR for OpenCV (dropping alpha channel)
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        return image

    elif message.HasField("yuv420"):
        # YUV420 format
        yuv_data = message.yuv420
        height, width = yuv_data.height, yuv_data.width

        # YUV420 has Y plane (height x width) + U and V planes (height/2 x width/2 each)
        y_size = height * width
        uv_size = (height // 2) * (width // 2)

        image_array = np.frombuffer(yuv_data.data, dtype=np.uint8)

        # Reshape YUV420 data
        y_plane = image_array[:y_size].reshape((height, width))
        u_plane = image_array[y_size:y_size + uv_size].reshape((height // 2, width // 2))
        v_plane = image_array[y_size + uv_size:].reshape((height // 2, width // 2))

        # Upsample U and V planes
        u_upsampled = cv2.resize(u_plane, (width, height), interpolation=cv2.INTER_LINEAR)
        v_upsampled = cv2.resize(v_plane, (width, height), interpolation=cv2.INTER_LINEAR)

        # Combine YUV planes
        yuv_image = np.stack([y_plane, u_upsampled, v_upsampled], axis=2)

        # Convert YUV to BGR
        image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)
        return image

    elif message.HasField("nv12"):
        # NV12 format (Y plane + interleaved UV plane)
        nv12_data = message.nv12
        height, width = nv12_data.height, nv12_data.width

        image_array = np.frombuffer(nv12_data.data, dtype=np.uint8)

        # NV12: Y plane (height x width) + UV plane (height/2 x width)
        y_size = height * width
        y_plane = image_array[:y_size].reshape((height, width))
        uv_plane = image_array[y_size:].reshape((height // 2, width))

        # Create full-size image for NV12 to BGR conversion
        nv12_image = np.zeros((height + height // 2, width), dtype=np.uint8)
        nv12_image[:height, :] = y_plane
        nv12_image[height:, :] = uv_plane

        # Convert NV12 to BGR
        image = cv2.cvtColor(nv12_image, cv2.COLOR_YUV2BGR_NV12)
        return image

    elif message.HasField("yuv422") or message.HasField("yuv444"):
        # Handle YUV422 and YUV444 formats
        if message.HasField("yuv422"):
            yuv_data = message.yuv422
        else:
            yuv_data = message.yuv444

        height, width = yuv_data.height, yuv_data.width

        # For YUV422/444, assume packed format
        image_array = np.frombuffer(yuv_data.data, dtype=np.uint8)
        image = image_array.reshape((height, width, 3))

        # Convert to BGR
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
        return image

    else:
        raise ValueError("Unsupported image format in ImageRawAny message")


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

    # Cache variables
    previous_orig_size = None
    previous_input_size = None
    scale_matrix = None

    def process_message(message: ImageRawAny):
        nonlocal previous_orig_size, previous_input_size, scale_matrix

        try:
            # Convert raw image to OpenCV format
            image = convert_raw_image_to_opencv(message)
            orig_height, orig_width = image.shape[:2]

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
                logging.info(f"Published {len(bboxes_2d.boxes)} bounding boxes")

        except Exception as e:
            logging.error(f"Error processing image: {e}")

    # Subscribe to incoming messages
    for sample in image_subscriber:
        try:
            message = image_encoder.decode(sample.payload.to_bytes())
            process_message(message)
        except Exception as e:
            logging.error(f"Error processing message: {e}")


if __name__ == "__main__":
    main()
