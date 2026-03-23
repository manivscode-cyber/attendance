import base64
import json
import sys
from io import BytesIO

import face_recognition
import numpy as np
from PIL import Image, ImageOps


def decode_image_from_base64(image_data):
    if "," in image_data:
        _, encoded = image_data.split(",", 1)
    else:
        encoded = image_data

    img_bytes = base64.b64decode(encoded)
    pil_image = Image.open(BytesIO(img_bytes))
    pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")
    return pil_image


def extract_face_data(image_data):
    if not image_data:
        return {
            "ok": False,
            "code": "missing_image",
            "message": "No image provided",
        }

    try:
        pil_image = decode_image_from_base64(image_data)
    except Exception:
        return {
            "ok": False,
            "code": "invalid_image",
            "message": "Invalid image data",
        }

    rgb_img = np.array(pil_image)
    if rgb_img.size == 0:
        return {
            "ok": False,
            "code": "invalid_image",
            "message": "Invalid image data",
        }

    scale = 1.0
    max_width = 960
    if pil_image.width > max_width:
        scale = max_width / float(pil_image.width)
        resized_height = int(pil_image.height * scale)
        resized = pil_image.resize((max_width, resized_height))
        processing_img = np.array(resized)
    else:
        processing_img = rgb_img

    def locate_faces(image, upsample=1):
        return face_recognition.face_locations(
            image,
            number_of_times_to_upsample=upsample,
            model="hog",
        )

    try:
        face_locations = locate_faces(processing_img, upsample=1)
        if len(face_locations) == 0:
            face_locations = locate_faces(processing_img, upsample=2)
        if len(face_locations) == 0:
            return {
                "ok": False,
                "code": "no_face",
                "message": (
                    "No face detected. Move closer, face the camera, "
                    "and use better lighting."
                ),
            }
        if len(face_locations) > 1:
            return {
                "ok": False,
                "code": "multiple_faces",
                "message": "Multiple faces detected. Please scan only one face.",
            }

        original_face_locations = face_locations
        if scale != 1.0:
            original_face_locations = [
                (
                    max(0, min(rgb_img.shape[0], int(top / scale))),
                    max(0, min(rgb_img.shape[1], int(right / scale))),
                    max(0, min(rgb_img.shape[0], int(bottom / scale))),
                    max(0, min(rgb_img.shape[1], int(left / scale))),
                )
                for top, right, bottom, left in face_locations
            ]

        encodings = face_recognition.face_encodings(processing_img, face_locations)
        if len(encodings) != 1 and processing_img is not rgb_img:
            encodings = face_recognition.face_encodings(
                rgb_img,
                original_face_locations,
            )
        if len(encodings) != 1:
            encodings = face_recognition.face_encodings(processing_img)
        if len(encodings) != 1 and processing_img is not rgb_img:
            encodings = face_recognition.face_encodings(rgb_img)
        if len(encodings) != 1:
            return {
                "ok": False,
                "code": "invalid_face",
                "message": "Please retry with one clear face in the frame.",
            }
    except Exception:
        return {
            "ok": False,
            "code": "processing_error",
            "message": "Please retry with one clear face in the frame.",
        }

    return {
        "ok": True,
        "encoding": [float(value) for value in encodings[0].tolist()],
        "location": [int(value) for value in original_face_locations[0]],
        "landmarks": None,
    }


def main():
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        json.dump(
            {
                "ok": False,
                "code": "invalid_payload",
                "message": "Invalid worker payload.",
            },
            sys.stdout,
        )
        return

    action = payload.get("action")
    if action != "extract_face":
        json.dump(
            {
                "ok": False,
                "code": "unsupported_action",
                "message": "Unsupported worker action.",
            },
            sys.stdout,
        )
        return

    json.dump(extract_face_data(payload.get("image_data")), sys.stdout)


if __name__ == "__main__":
    main()
