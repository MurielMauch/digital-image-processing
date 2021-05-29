import cv2
import numpy as np

import matplotlib.pyplot as plt

original_img = cv2.imread(f'imagens/input/objetos3.png')
# plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
# plt.show()

new_image = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
# plt.imshow(new_image, cmap='gray')
# plt.show()

ret, bw = cv2.threshold(new_image, 200, 255, cv2.THRESH_BINARY)
plt.imshow(bw, cmap='gray')
plt.show()

new_image_2 = cv2.blur(bw, (3, 3))

canny_output = cv2.Canny(new_image_2, 200, 255, 3)
plt.imshow(canny_output, cmap='gray')
plt.show()

contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

desenho_contornos = np.ones((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8) * 255
contornos_numerados = np.ones((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8) * 255

pequena = 0
media = 0
grande = 0
areas = []

for i in range(len(contours)):
    contorno = contours[i]
    area = cv2.contourArea(contorno)
    if area < 1500:
        pequena += 1
    elif area >= 1500 and area < 3000:
        media += 1
    elif area > 3000:
        grande += 1
    areas.append(area)
    perimetro = cv2.arcLength(contorno, True)

    hull = cv2.convexHull(contorno)
    hull_area = cv2.contourArea(hull)
    solidez = area / hull_area

    # compute the center of the contour
    M = cv2.moments(contorno)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    cv2.drawContours(desenho_contornos, contours, i, [255, 0, 0], 1, cv2.LINE_8, hierarchy, 0)

    cv2.drawContours(contornos_numerados, contours[i], -1, [0, 0, 0], 1, cv2.LINE_8)
    cv2.circle(contornos_numerados, (cX, cY), 7, (255, 255, 255), -1)
    cv2.putText(contornos_numerados, f"{i}", (cX - 2, cY + 2), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                color=(255, 0, 0), thickness=1)

    elipse = cv2.fitEllipse(contorno)
    centro, eixo, orientacao = elipse

    eixo_a = max(eixo)
    eixo_b = min(eixo)
    excentricidade = np.sqrt(1 - (eixo_b / eixo_a) ** 2)

    print(
        f'Região {i}: área: {area} perimetro: {perimetro:.4f} excentricidade: {excentricidade:.4f} solidez: {solidez}')

plt.imshow(desenho_contornos, cmap='gray')
plt.show()

plt.imshow(contornos_numerados, cmap='gray')
plt.show()

print(f'Numero de regioes pequenas: {pequena}\nNumero de regioes medias: {media}\nNumero de regioes grandes: {grande}')

entradas = np.array(areas)
plt.hist(entradas, bins=[0, 1500, 3000, 4500])
plt.show()
