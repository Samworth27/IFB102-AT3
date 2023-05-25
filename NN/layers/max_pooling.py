from .layer import Layer
import numpy as np


class MaxPooling:
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def patches_generator(self, image):
        output_height = image.shape[0] // self.kernel_size
        output_width = image.shape[1] // self.kernel_size
        self.image = image

        for h in range(output_height):
            for w in range(output_width):
                patch = image[
                    (h * self.kernel_size) : (h * self.kernel_size + self.kernel_size),
                    (w * self.kernel_size) : (w * self.kernel_size + self.kernel_size),
                ]
                yield patch, h, w

    def forwards(self, image):
        image_height, image_width, num_kernels = image.shape
        max_pooling_output = np.zeros(
            (
                image_height // self.kernel_size,
                image_width // self.kernel_size,
                num_kernels,
            )
        )
        for patch, h, w in self.patches_generator(image):
            max_pooling_output[h, w] = np.amax(patch, axis=(0, 1))
        return max_pooling_output

    def backwards(self, dE_dY, learning_rate):
        dE_dk = np.zeros(self.image.shape)
        for patch, h, w in self.patches_generator(self.image):
            image_h, image_w, num_kernels = patch.shape
            max_val = np.amax(patch, axis=(0, 1))

            for idx_h in range(image_h):
                for idx_w in range(image_w):
                    for idx_k in range(num_kernels):
                        if patch[idx_h, idx_w, idx_k] == max_val[idx_k]:
                            dE_dk[
                                h * self.kernel_size + idx_h,
                                w * self.kernel_size + idx_w,
                                idx_k,
                            ] = dE_dY[h, w, idx_k]
            return dE_dk
        
    def dropout(self, dropout_rate):
        pass
