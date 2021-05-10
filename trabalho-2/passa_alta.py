import cv2
import numpy as np
from utils import HandlingImages
import sys

image_handler = HandlingImages()
images = image_handler.get_image_list()

for image in images:
    for radius in range(10, 101, 20):
        original_img = cv2.imread(f'images/input/{image}.png', 0)
        image_handler.add_img_to_plot(original_img, 'Original', 1, 4, 1)
        original_img_min = original_img.min()
        original_img_max = original_img.max()

        # aplicando a transformada de fourier
        magnitude_spectrum, fft_shift = image_handler.apply_fourier(original_img)
        image_handler.add_img_to_plot(magnitude_spectrum, 'Espectro de magnitude', 1, 4, 2)

        # obtendo o centro da imagem e criando a máscara
        rows, cols = magnitude_spectrum.shape
        central_row, central_col = int(rows / 2), int(cols / 2)
        mask = np.full((rows, cols), 1, np.uint8)
        img_center = (central_row, central_col)
        mask = cv2.circle(mask, img_center, radius, (0, 0, 0), -1)

        # aplicando a máscara no espectro de magnitude
        espectro_com_mascara = magnitude_spectrum * mask
        image_handler.add_img_to_plot(espectro_com_mascara, 'Espectro de magnitude com máscara', 1, 4, 3)

        # criando a máscara novamente aumentando uma dimensão
        mask = np.full((rows, cols, 2), 1, np.uint8)
        img_center = (central_row, central_col)
        mask = cv2.circle(mask, img_center, radius, (0, 0, 0), -1)
        # aplicando a inversa
        fshift = fft_shift * mask
        img_back = image_handler.apply_inverse_fourier(fshift, original_img_min, original_img_max)
        image_handler.add_img_to_plot(img_back, 'Imagem com máscara', 1, 4, 4)

        image_handler.plot_images()

        cv2.imwrite(f'images/outputs/passa_alta/{image}/{radius}-1-original.png', original_img)
        cv2.imwrite(f'images/outputs/passa_alta/{image}/{radius}-2-espectro-magnitude.png', magnitude_spectrum)
        cv2.imwrite(f'images/outputs/passa_alta/{image}/{radius}-3-espectro-magnitude-com-mascara.png',
                    espectro_com_mascara)
        cv2.imwrite(f'images/outputs/passa_alta/{image}/{radius}-4-inversa.png', img_back)
