"""Fail fast unless TensorFlow can execute a real operation on a GPU."""

from __future__ import annotations

import tensorflow as tf


def main() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("TensorFlow did not detect a GPU")

    with tf.device("/GPU:0"):
        left = tf.random.uniform((2048, 2048))
        right = tf.random.uniform((2048, 2048))
        product = tf.matmul(left, right)
        sample = float(product[0, 0].numpy())

    if "GPU:0" not in product.device:
        raise RuntimeError("Matrix multiplication did not execute on GPU: {}".format(product.device))

    build = tf.sysconfig.get_build_info()
    print("TensorFlow: {}".format(tf.__version__))
    print("GPU: {}".format(gpus[0].name))
    print("CUDA: {}".format(build.get("cuda_version")))
    print("cuDNN: {}".format(build.get("cudnn_version")))
    print("Matmul device: {}".format(product.device))
    print("Matmul sample: {:.6f}".format(sample))


if __name__ == "__main__":
    main()
