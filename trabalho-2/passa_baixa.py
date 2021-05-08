import cv2
import numpy as np
import sys

from utils import HandlingImages

show_preliminary_result = bool(sys.argv[1])

image_handler = HandlingImages()
images = image_handler.get_image_list()

for image in images:
    for radius in range(30, 91, 30):
        original_img = cv2.imread(f'images/input/{image}.png', 0)
        image_handler.add_img_to_plot(original_img, 'Original', 1, 4, 1)

        # applying fourier transform
        magnitude_spectrum, dft_shift = image_handler.apply_fourier(original_img)
        image_handler.add_img_to_plot(magnitude_spectrum, 'Magnitude Spectrum', 1, 4, 2)

        # applying ther mask in the magnitude spectrum
        # getting image center
        rows, cols = magnitude_spectrum.shape
        central_row, central_col = int(rows / 2), int(cols / 2)
        # creating mask varying the radius

        mask = np.zeros((rows, cols), np.uint8)
        img_center = (central_row, central_col)
        mask = cv2.circle(mask, img_center, radius, (1, 1, 1), -1)

        # applying the mask in the magnitude spectrum
        spectrum_with_mask = magnitude_spectrum * mask
        image_handler.add_img_to_plot(spectrum_with_mask, 'Magnitude Spectrum x Mask', 1, 4, 3)

        # transforming the image back
        mask = np.zeros((rows, cols, 2), np.uint8)
        img_center = (central_row, central_col)
        mask = cv2.circle(mask, img_center, radius, (1, 1, 1), -1)

        # apply mask and inverse FFT
        fshift = dft_shift * mask
        img_back = image_handler.apply_inverse_fourier(fshift)
        image_handler.add_img_to_plot(img_back, 'Reverse Image with Mask', 1, 4, 4)

        if show_preliminary_result:
            image_handler.plot_images()

        cv2.imwrite(f'images/outputs/passa_baixa/{image}/{radius}-1-original.png', original_img)
        cv2.imwrite(f'images/outputs/passa_baixa/{image}/{radius}-2-espectro-magnitude.png', magnitude_spectrum)
        cv2.imwrite(f'images/outputs/passa_baixa/{image}/{radius}-3-espectro-magnitude-com-mascara.png',
                    spectrum_with_mask)
        cv2.imwrite(f'images/outputs/passa_baixa/{image}/{radius}-4-inversa.png', img_back)
