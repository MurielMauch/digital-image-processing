import cv2
import numpy as np
import sys

from utils import HandlingImages

image_handler = HandlingImages()
images = image_handler.get_image_list()

for image in images:
    for radius in range(30, 91, 30):
        original_img = cv2.imread(f'images/input/{image}.png', 0)
        image_handler.add_img_to_plot(original_img, 'Original', 1, 4, 1)
        original_img_min = original_img.min()
        original_img_max = original_img.max()

        # aplicando a transformada de fourier
        magnitude_spectrum, fft_shift = image_handler.apply_fourier(original_img)
        image_handler.add_img_to_plot(magnitude_spectrum, 'Magnitude Spectrum', 1, 4, 2)

        # obtendo o centro da imagem
        rows, cols = magnitude_spectrum.shape
        central_row, central_col = int(rows / 2), int(cols / 2)
        # criando a máscara
        mask = np.zeros((rows, cols), np.uint8)
        img_center = (central_row, central_col)
        mask = cv2.circle(mask, img_center, radius, (1, 1, 1), -1)

        # aplicando a máscara no espectro de magnitude
        spectrum_with_mask = magnitude_spectrum * mask
        image_handler.add_img_to_plot(spectrum_with_mask, 'Magnitude Spectrum x Mask', 1, 4, 3)

        # criando a máscara novamente aumentando uma dimensão
        mask = np.zeros((rows, cols, 2), np.uint8)
        img_center = (central_row, central_col)
        mask = cv2.circle(mask, img_center, radius, (1, 1, 1), -1)

        # aplicando a máscara e obtendo a inversa da transformada
        fshift = fft_shift * mask
        img_back = image_handler.apply_inverse_fourier(fshift, original_img_min, original_img_max)
        image_handler.add_img_to_plot(img_back, 'Reverse Image with Mask', 1, 4, 4)

        image_handler.plot_images()

        cv2.imwrite(f'images/outputs/passa_baixa/{image}/{radius}-1-original.png', original_img)
        cv2.imwrite(f'images/outputs/passa_baixa/{image}/{radius}-2-espectro-magnitude.png', magnitude_spectrum)
        cv2.imwrite(f'images/outputs/passa_baixa/{image}/{radius}-3-espectro-magnitude-com-mascara.png',
                    spectrum_with_mask)
        cv2.imwrite(f'images/outputs/passa_baixa/{image}/{radius}-4-inversa.png', img_back)
