import cv2
import numpy as np
from matplotlib import pyplot as plt


class HandlingImages:
    def __init__(self):
        self.images_to_plot = ['baboon', 'butterfly', 'city', 'house', 'seagull']
        self.plot = plt
        self.row, self.column = None, None

    def get_image_list(self):
        return self.images_to_plot

    def create_subplot(self):
        self.plot.figure()

    def add_img_to_plot(self, img, title, row, column, position):
        self.plot.subplot(row, column, position)
        self.plot.imshow(img, cmap='gray')
        self.plot.title(title)

    def plot_images(self):
        self.plot.show()

    def apply_fourier(self, img):
        # aplicando a transformada de fourier
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        # realizando o shift para o centro da imagem
        dft_shift = np.fft.fftshift(dft)
        # obtendo o spectro de magnitude
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

        return magnitude_spectrum, dft_shift

    def apply_inverse_fourier(self, dft_shift):
        f_ishift = np.fft.ifftshift(dft_shift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        norm_image = cv2.normalize(img_back, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)

        return norm_image
