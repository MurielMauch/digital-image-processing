import cv2
import numpy as np
import sys

import matplotlib.pyplot as plt

for image in ['objetos1', 'objetos2', 'objetos3']:
    img_name = image
    print('\n\n#########################################################################################')
    print(f'Identificando contornos e propriedades da imagem {img_name}\n')

    # lendo a imagem original
    original_img = cv2.imread(f'imagens/input/{img_name}.png')
    if original_img is None:
        print('Falha ao ler a imagem')
        sys.exit(1)

    # transformando imagem em monocromatica
    imagem_monocromatica = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

    # binarizando a imagem monocromatia e salvando em arquivo
    ret, bw = cv2.threshold(imagem_monocromatica, 200, 255, cv2.THRESH_BINARY)

    cv2.imwrite(f'imagens/outputs/{img_name}-monocromatica.png', imagem_monocromatica)

    # removendo 'noise' e detectando as bordas da imagem
    blurred_image = cv2.blur(bw, (3, 3))
    canny_image = cv2.Canny(blurred_image, 200, 255, 3)

    # encontrando os contornos
    contornos, hierarchy = cv2.findContours(canny_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # desenhando os contornos
    desenho_contornos = np.ones((canny_image.shape[0], canny_image.shape[1], 3), dtype=np.uint8) * 255
    contornos_numerados = np.ones((canny_image.shape[0], canny_image.shape[1], 3), dtype=np.uint8) * 255

    # inicializando os contadores das areas
    pequena = 0
    media = 0
    grande = 0
    areas = []

    # para cada contorno encontrado, vamos extrair suas propriedades
    # area, solidez, centroide
    for i in range(len(contornos)):
        contorno = contornos[i]

        # identificando a area, incrementando o contador de acordo com o valor da area
        area = cv2.contourArea(contorno)
        if area < 1500:
            pequena += 1
        elif area >= 1500 and area < 3000:
            media += 1
        elif area > 3000:
            grande += 1
        areas.append(area)

        # calculando o perimetro
        perimetro = cv2.arcLength(contorno, True)

        # calculando a solidez
        hull_convexo = cv2.convexHull(contorno)
        area_hull = cv2.contourArea(hull_convexo)
        solidez = area / area_hull

        # desenhando os contornos
        cv2.drawContours(desenho_contornos, contornos, i, [255, 0, 0], 1, cv2.LINE_8, hierarchy, 0)

        # calculando exentricidade
        elipse = cv2.fitEllipse(contorno)

        # coordenadas do centro, eixos da elipse e angulo da orientacao
        (x, y), (eixo_a, eixo_b), angle = cv2.fitEllipse(contorno)
        x, y = round(x), round(y)
        a = eixo_b / 2
        b = eixo_a / 2
        excentricidade = np.sqrt(pow(a, 2) - pow(b, 2))
        excentricidade = round(excentricidade / a, 2)

        # numerando os contornos
        cv2.drawContours(contornos_numerados, contornos[i], -1, [0, 0, 0], 1, cv2.LINE_8)
        cv2.circle(contornos_numerados, (x, y), 7, (255, 255, 255), -1)
        cv2.putText(original_img, f"{i}", (x, y), cv2.FONT_HERSHEY_PLAIN, fontScale=1,
                    color=(0, 0, 0), thickness=1)


        print(
            f'Região {i}: área: {area} perimetro: {perimetro:.7f} excentricidade: {excentricidade:.7f} solidez: {solidez:.7f}')

    cv2.imwrite(f'imagens/outputs/{img_name}-contornos.png', desenho_contornos)
    cv2.imwrite(f'imagens/outputs/{img_name}-contornos-numerados.png', original_img)

    print(
        f'\nNumero de regioes pequenas: {pequena}\nNumero de regioes medias: {media}\nNumero de regioes grandes: {grande}')

    entradas = np.array(areas)
    plt.hist(entradas, bins=[0, 1500, 3000, 4500])
    plt.ylabel('Número de contornos')
    plt.xlabel('Área')
    plt.savefig(f'imagens/outputs/{img_name}-histograma.png')
    plt.clf()
