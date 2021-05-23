import cv2
import numpy as np

import matplotlib.pyplot as plt

original_img = cv2.imread(f'imagens/input/objetos1.png')
# plt.imshow('Input Image', cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
# plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
# plt.show()

new_image = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
# plt.imshow(new_image, cmap='gray')
# plt.show()

canny_output = cv2.Canny(new_image, 30, 200)
contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

desenho_contornos = np.ones((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8) * 255
contornos_numerados = np.ones((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8) * 255

pequena = 0
media = 0
grande = 0
for i in range(len(contours)):
    contorno = contours[i]
    area = cv2.contourArea(contorno)
    if area < 1500:
        pequena += 1
    elif area >= 1500 and area < 3000:
        media += 1
    elif area > 3000:
        grande += 1
    perimetro = cv2.arcLength(contorno, True)

    hull = cv2.convexHull(contorno)
    hull_area = cv2.contourArea(hull)
    solidez = float(area) / hull_area

    # compute the center of the contour
    M = cv2.moments(contorno)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    cv2.drawContours(desenho_contornos, contours, i, [255, 0, 0], 1, cv2.LINE_8, hierarchy, 0)

    cv2.drawContours(contornos_numerados, contours[i], -1, [0, 0, 0], 1, cv2.LINE_8)
    cv2.circle(contornos_numerados, (cX, cY), 7, (255, 255, 255), -1)
    cv2.putText(contornos_numerados, f"{i}", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    print(
        f'Região f{i}: área: {area} perimetro: {perimetro} excentricidade: {0} solidez: {solidez} centro: ({cX},{cY})')

# plt.imshow(desenho_contornos, cmap='gray')
# plt.show()

# plt.imshow(contornos_numerados, cmap='gray')
# plt.show()

print(f'Numero de regioes pequenas: {pequena}\nNumero de regioes medias: {media}\nNumero de regioes grandes: {grande}')

histograma = np.histogram([pequena, media, grande])
plt.hist(histograma)
plt.show()
