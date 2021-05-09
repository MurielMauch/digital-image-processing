import cv2
from utils import HandlingImages

image_handler = HandlingImages()
images = image_handler.get_image_list()

for image in images:
    original_img = cv2.imread(f'images/input/{image}.png', 0)
    image_handler.add_img_to_plot(original_img, 'Original', 1, 3, 1)
    original_img_min = original_img.min()
    original_img_max = original_img.max()

    # applying the fourier transform
    magnitude_spectrum, dft_shift = image_handler.apply_fourier(original_img)
    image_handler.add_img_to_plot(magnitude_spectrum, 'Mask', 1, 3, 2)

    # transforming the image back to the original
    img_inversa = image_handler.apply_inverse_fourier(dft_shift, original_img_min, original_img_max)
    image_handler.add_img_to_plot(img_inversa, 'Reverse', 1, 3, 3)

    image_handler.plot_images()

    cv2.imwrite(f'images/outputs/visualizacao/{image}-espectro-magnitude.png', magnitude_spectrum)
    cv2.imwrite(f'images/outputs/visualizacao/{image}-inversa-original.png', img_inversa)
