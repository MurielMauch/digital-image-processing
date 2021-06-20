import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.spatial.distance import euclidean
from skimage import io, color, img_as_ubyte
from skimage.feature import greycomatrix, greycoprops
from skimage.feature import local_binary_pattern
from prettytable import PrettyTable

imagens = ['textura1', 'textura2', 'textura3', 'textura4', 'textura5', 'textura6', 'textura7', 'textura8']
rgba = ['textura5', 'textura6', 'textura7', 'textura8']
histogramas = []
constrastes = []

for nome_imagem in imagens:
    imagem_entrada = io.imread(f'imagens/input/{nome_imagem}.png')

    imagem_entrada_min = imagem_entrada.min()
    imagem_entrada_max = imagem_entrada.max()

    # transformando imagem em monocromatica e escrevendo em disco
    if nome_imagem not in rgba:
        imagem_monocromatica = color.rgb2gray(imagem_entrada)
    else:
        imagem_monocromatica = color.rgb2gray(color.rgba2rgb(imagem_entrada))

    imagem_monocromatica_normalizada = cv2.normalize(imagem_monocromatica, None, alpha=imagem_entrada_min,
                                                     beta=imagem_entrada_max, norm_type=cv2.NORM_MINMAX,
                                                     dtype=cv2.DFT_COMPLEX_OUTPUT)
    io.imsave(f'imagens/output/{nome_imagem}-monocromatica.png', imagem_monocromatica_normalizada)

    # calculando lbp
    lbp = local_binary_pattern(imagem_monocromatica_normalizada, 8, 1)
    io.imsave(f'imagens/output/{nome_imagem}-lbp.png', lbp.astype(np.uint8))

    # criando o histograma e plottando
    bins = int(lbp.max()) + 1
    hist, _ = np.histogram(lbp, bins=bins, density=True)
    histogramas.append(hist)
    plt.plot(hist)
    plt.savefig(f'imagens/output/{nome_imagem}-histograma.png')

    # calculando a glcm
    # 0 graus, 45 graus, 90 graus
    angulos = [0, np.pi / 4, np.pi / 2]
    glcm = greycomatrix(img_as_ubyte(imagem_monocromatica_normalizada),
                        distances=[1], angles=angulos, symmetric=True, normed=True)

    aux = greycoprops(glcm, 'contrast').tolist()[0]
    aux.insert(0, nome_imagem)
    constrastes.append(aux)
    plt.clf()

print('Contraste')
y = PrettyTable()
y.field_names = ['nome da imagem', '0 graus', '45 graus', '90 graus']
for index in constrastes:
    y.add_row(index)
print(y)

print('\n\nDistancias')
x = PrettyTable()
imagens.insert(0, 'nome da imagem')
x.field_names = imagens
count = 1
for hist in histogramas:
    dist = []
    for i in range(len(histogramas)):
        dist.append(euclidean(hist, histogramas[i]))
    dist.insert(0, f'textura{count}')
    count += 1
    x.add_row(dist)
print(x)
