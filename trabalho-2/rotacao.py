import cv2

from utils import HandlingImages

image_handler = HandlingImages()
images = image_handler.get_image_list()

for image in images:
    original_img = cv2.imread(f'images/input/{image}.png', 0)
    image_handler.add_img_to_plot(original_img, 'Original', 1, 4, 1)

    magnitude_spectrum, dft_shift = image_handler.apply_fourier(original_img)
    image_handler.add_img_to_plot(magnitude_spectrum, 'Original Magnitude Spectrum', 1, 4, 2)

    rows, cols = original_img.shape

    t_image = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 0.7)
    dst = cv2.warpAffine(original_img, t_image, (cols, rows))

    image_handler.add_img_to_plot(dst, 'Translated Image', 1, 4, 3)

    t_magnitude_spectrum, dft_shift = image_handler.apply_fourier(dst)

    image_handler.add_img_to_plot(t_magnitude_spectrum, 'Translated Image Magnitude Spectrum', 1, 4, 4)

    image_handler.plot_images()

    cv2.imwrite(f'images/outputs/rotacao/{image}-original.png', original_img)
    cv2.imwrite(f'images/outputs/rotacao/{image}-espectro-magnitude.png', magnitude_spectrum)
    cv2.imwrite(f'images/outputs/rotacao/{image}-transladada.png', dst)
    cv2.imwrite(f'images/outputs/rotacao/{image}-transladada-espectro-magnitude.png', t_magnitude_spectrum)

