import cv2
from flask import url_for
import numpy as np
import tensorflow as tf
import hashlib
from flask_queue_sse import ServerSentEvents


def format_sse(data: str, event=None) -> str:
    """Formats a string and an event name in order to follow the event stream convention.

    >>> format_sse(data=json.dumps({'abc': 123}), event='Jackson 5')
    'event: Jackson 5\\ndata: {"abc": 123}\\n\\n'

    """
    msg = f"data: {data}\n\n"
    if event is not None:
        msg = f"event: {event}\n{msg}"
    return msg


class ServerSentEventsCallback(tf.keras.callbacks.Callback):
    def __init__(self, announcer: ServerSentEvents, batch_length: int):
        super().__init__()
        self.announcer = announcer
        self.batch_length = batch_length

    def on_predict_batch_end(self, batch, logs=None):
        announce_progress(
            self.announcer,
            {"size": int(self.params.get("steps")), "current": batch},
        )


def announce_progress(announcer, msg):
    announcer.send(msg)


def get_patches(file_name, patch_size, crop_sizes):
    """This functions creates and return patches of given image with a specified patch_size"""
    image = cv2.imread(file_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, channels = image.shape
    patches = []
    for crop_size in crop_sizes:  # We will crop the image to different sizes
        crop_h, crop_w = int(height * crop_size), int(width * crop_size)
        image_scaled = cv2.resize(
            image, (crop_w, crop_h), interpolation=cv2.INTER_CUBIC
        )
        for i in range(0, crop_h - patch_size + 1, patch_size):
            for j in range(0, crop_w - patch_size + 1, patch_size):
                x = image_scaled[
                    i : i + patch_size, j : j + patch_size
                ]  # This gets the patch from the original image with size patch_size x patch_size
                patches.append(x)
    return patches


def create_image_from_patches(patches, image_shape):
    """This function takes the patches of images and reconstructs the image"""
    image = np.zeros(
        image_shape
    )  # Create a image with all zeros with desired image shape
    patch_size = patches.shape[1]
    p = 0
    for i in range(0, image.shape[0] - patch_size + 1, patch_size):
        for j in range(0, image.shape[1] - patch_size + 1, patch_size):
            image[i : i + patch_size, j : j + patch_size] = patches[
                p
            ]  # Assigning values of pixels from patches to image
            p += 1
    return np.array(image)


def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))


def PSNR(y_true, y_pred):
    """Compute the peak signal-to-noise ratio, measures quality of image."""
    psnr_value = tf.image.psnr(y_true, y_pred, max_val=1.0)[0]
    return psnr_value


model = tf.keras.models.load_model(
    "ridnet", custom_objects={"ssim_loss": ssim_loss, "PSNR": PSNR}
)


async def predict(
    id: str, image_path: str, save_path: str, announcer: ServerSentEvents
):
    patches = get_patches(image_path, 40, [1])
    test_image = cv2.imread(image_path)

    patches = np.array(patches)
    patches = patches.astype("float32") / 255.0
    patches_noisy = patches
    patches_noisy = tf.clip_by_value(
        patches_noisy, clip_value_min=0.0, clip_value_max=1.0
    )
    noisy_image = create_image_from_patches(patches, test_image.shape) / 255.0
    denoised_patches = model.predict(
        patches_noisy, callbacks=[ServerSentEventsCallback(announcer, len(patches))]
    )
    print("prediction complete")
    denoised_patches = tf.clip_by_value(
        denoised_patches, clip_value_min=0.0, clip_value_max=1.0
    )

    # Creating entire denoised image from denoised patches
    denoised_image = create_image_from_patches(denoised_patches, test_image.shape)
    cv2.imwrite(
        save_path,
        cv2.cvtColor((255 * denoised_image).astype("uint8"), cv2.COLOR_RGB2BGR),
    )
    print("image written")
    data = {
        "id": id,
        "filepath": "/denoised/" + id + ".jpg",
        "realpath": "/uploads/" + id + ".jpg",
    }
    print("data primed")
    announcer.send(data, event="end")
    print("message send")
    return patches, denoised_patches, noisy_image, denoised_image


def hash_file(filename):
    """ "This function returns the SHA-1 hash
    of the file passed into it"""

    # make a hash object
    h = hashlib.sha1()

    # open file for reading in binary mode
    with open(filename, "rb") as file:

        # loop till the end of the file
        chunk = 0
        while chunk != b"":
            # read only 1024 bytes at a time
            chunk = file.read(1024)
            h.update(chunk)

    # return the hex representation of digest
    return h.hexdigest()
