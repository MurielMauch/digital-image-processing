import cv2
import numpy as np

from utils import HandlingImages

image_handler = HandlingImages()
images = image_handler.get_image_list()

for image in images:
    # lendo as imagens e obtendo os valores minimo e maximo
    original_img = cv2.imread(f'images/input/{image}.png', 0)
    image_handler.add_img_to_plot(original_img, 'Original', 1, 5, 1)
    cv2.imwrite(f'images/outputs/compressao/{image}/original.png', original_img)
    original_img_min = original_img.min()
    original_img_max = original_img.max()

    # aplicado a transformada
    espectro_magnitude, fft = image_handler.apply_fourier(original_img)

    # ordenamos a transformada do maior para o menor
    fft_sorted = np.sort(np.abs(fft.reshape(-1)))

    contador = 2  # contador inicia em dois pois essa será a posição da primeira iteração no plot do grafico
    for radius in (0.1, 0.05, 0.01, 0.002):
        # obtemos o threshold de nossa imagem
        thresh = fft_sorted[int(np.floor((1 - radius) * len(fft_sorted)))]
        # definimos todos os valores acima desse threshold como 1
        # os demais como zero
        ind = np.abs(fft) > thresh
        atlow = fft * ind

        # aplicamos a inversa da transformada
        alow = image_handler.apply_inverse_fourier(atlow, original_img_min, original_img_max)

        image_handler.add_img_to_plot(alow, f'compressed with {radius}', 1, 5, contador)
        cv2.imwrite(f'images/outputs/compressao/{image}/compressed_{radius}.png', alow)

        contador += 1

    image_handler.plot_images()
