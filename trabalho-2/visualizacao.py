import cv2
from utils import HandlingImages

image_handler = HandlingImages()
images = image_handler.get_image_list()

for image in images:
    # lemos a imagem de entrada
    original_img = cv2.imread(f'images/input/{image}.png', 0)
    # adicionamos ela para ser plottada
    image_handler.add_img_to_plot(original_img, 'Original', 1, 3, 1)
    # identificamos o valor minimo e maximo da imagem
    original_img_min = original_img.min()
    original_img_max = original_img.max()

    # aplicamos a transformada de fourier e a adicionamos para plottagem
    magnitude_spectrum, fft_shift = image_handler.apply_fourier(original_img)
    image_handler.add_img_to_plot(magnitude_spectrum, 'Mask', 1, 3, 2)

    # transformamos a imagem obtido de volta para o dominio espacial
    img_inversa = image_handler.apply_inverse_fourier(fft_shift, original_img_min, original_img_max)
    image_handler.add_img_to_plot(img_inversa, 'Reverse', 1, 3, 3)

    image_handler.plot_images()

    cv2.imwrite(f'images/outputs/visualizacao/{image}-espectro-magnitude.png', magnitude_spectrum)
    cv2.imwrite(f'images/outputs/visualizacao/{image}-inversa-original.png', img_inversa)
