import cv2
import numpy as np

from utils import HandlingImages

image_handler = HandlingImages()
images = image_handler.get_image_list()

for image in images:
    for radius in range(10, 21, 2):
        original_img = cv2.imread(f'images/input/{image}.png', 0)
        image_handler.add_img_to_plot(original_img, 'Original Image', 1, 4, 1)

        magnitude_spectrum, dft_shift = image_handler.apply_fourier(original_img)
        image_handler.add_img_to_plot(magnitude_spectrum, 'Espectro de magnitude', 1, 4, 2)

        outer_radius = radius * 7

        rows, cols = magnitude_spectrum.shape
        central_row, central_col = int(rows / 2), int(cols / 2)
        mask = np.zeros((rows, cols), np.uint8)
        img_center = (central_row, central_col)
        mask = cv2.circle(mask, img_center, outer_radius, (1, 1, 1), -1)
        mask = cv2.circle(mask, img_center, radius, (0, 0, 0), -1)

        espectro_com_mascara = magnitude_spectrum * mask
        image_handler.add_img_to_plot(espectro_com_mascara, 'Espectro de magnitude com máscara', 1, 4, 3)

        mask = np.zeros((rows, cols, 2), np.uint8)
        img_center = (central_row, central_col)
        mask = cv2.circle(mask, img_center, outer_radius, (1, 1, 1), -1)
        mask = cv2.circle(mask, img_center, radius, (0, 0, 0), -1)

        fshift = dft_shift * mask
        img_back = image_handler.apply_inverse_fourier(fshift)
        image_handler.add_img_to_plot(img_back, 'Imagem com máscara', 1, 4, 4)

        # image_handler.plot_images()

        cv2.imwrite(f'images/outputs/passa_faixa/{image}/{radius}-original-{outer_radius}.png', original_img)
        cv2.imwrite(f'images/outputs/passa_faixa/{image}/{radius}-espectro-magnitude-{outer_radius}.png',
                    magnitude_spectrum)
        cv2.imwrite(f'images/outputs/passa_faixa/{image}/{radius}-espectro-magnitude-com-mascara-{outer_radius}.png',
                    espectro_com_mascara)
        cv2.imwrite(f'images/outputs/passa_faixa/{image}/{radius}-inversa.png-{outer_radius}', img_back)
