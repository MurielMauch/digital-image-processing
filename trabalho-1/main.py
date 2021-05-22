import matplotlib
import numpy as np
import sys
import imageio

import filters

from matplotlib import pyplot as plt

matplotlib.use('TkAgg')

#######################################################################################################################
# seção para validar as entradas e abrir a imagem
#######################################################################################################################

if len(sys.argv) < 3:
    print('Você não especificou os argumentos de entradas necessário. Por favor, passe os seguintes argumentos nessa '
          'ordem \n\n'
          'python main.py nome_da_imagem.png filtro pattern\n\n'
          'Os valores aceitos são:\n'
          'nome_da_imagem = [baboon.png, monalisa.png, peppers.png, watch.png, butterfly.png, city.png, seagull.png]\n'
          'filtro = [h1 | h2 | h1_and_h2 | h3 | h4 | h5 | h6 | h7 | h8 | h9 | 1a | 1b]\n'
          'pattern = [mono | colored]\n'
          )
    sys.exit(1)

img_name = sys.argv[1]  # arg1
filter_to_apply = sys.argv[2]  # arg2
image_pattern = sys.argv[3]  # arg3

if img_name not in ['baboon.png', 'monalisa.png', 'peppers.png', 'watch.png', 'butterfly.png', 'city.png',
                    'seagull.png']:
    print('Nome da imagem é inválido: {}'.format(img_name))
    sys.exit(1)
elif filter_to_apply not in ['h1', 'h2', 'h1_and_h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9', '1a', '1b']:
    print('Filtro inválido: {}'.format(filter_to_apply))
    sys.exit(1)
elif image_pattern not in ['mono', 'colored']:
    print('Pattern inválido: {}'.format(image_pattern))
    sys.exit(1)

# open image file and stores it in a numpy array
try:
    original_image = imageio.imread(f'input_data/{image_pattern}/{img_name}')
except Exception as e:
    print("WARNING: could not open image file: {}".format(e))
    sys.exit(1)

#######################################################################################################################
# seção para adicionar moldura na imagem
#######################################################################################################################

if image_pattern == 'mono':
    leng_of_img = len(original_image[0]) + 2

    # adicionando uma moldura na imagem monocromática
    img_bw_moldura = np.zeros(shape=(leng_of_img, leng_of_img), dtype=original_image.dtype)

    original_img_height, original_img_width = original_image.shape

    # compute center offset
    x_center = (leng_of_img - original_img_height) // 2
    y_center = (leng_of_img - original_img_width) // 2

    # copy img into center of result image
    img_bw_moldura[y_center:y_center + original_img_height, x_center:x_center + original_img_width] = original_image

    height, width = img_bw_moldura.shape[:2]

#######################################################################################################################
# seção para aplicar os filtros, mostrar em tela e salvar a imagem
#######################################################################################################################
new_image = None

if filter_to_apply == 'h1':
    new_image = filters.apply_filter_h1(img_bw_moldura, height, width)  # noqa
elif filter_to_apply == 'h2':
    new_image = filters.apply_filter_h2(img_bw_moldura, height, width)  # noqa
elif filter_to_apply == 'h1_and_h2':
    new_image = filters.apply_filters_h1_and_h2(img_bw_moldura, height, width)  # noqa
elif filter_to_apply == 'h3':
    new_image = filters.apply_filtro_h3(img_bw_moldura, height, width)  # noqa
elif filter_to_apply == 'h4':
    new_image = filters.apply_filter_h4(img_bw_moldura, height, width)  # noqa
elif filter_to_apply == 'h5':
    new_image = filters.apply_filter_h5(img_bw_moldura, height, width)  # noqa
elif filter_to_apply == 'h6':
    new_image = filters.apply_filter_h6(img_bw_moldura, height, width)  # noqa
elif filter_to_apply == 'h7':
    new_image = filters.apply_filter_h7(img_bw_moldura, height, width)  # noqa
elif filter_to_apply == 'h8':
    new_image = filters.apply_filter_h8(img_bw_moldura, height, width)  # noqa
elif filter_to_apply == 'h9':
    new_image = filters.apply_filter_h9(img_bw_moldura, height, width)  # noqa
elif filter_to_apply == '1a':
    new_image = filters.apply_filter_1a(original_image)
elif filter_to_apply == '1b':
    new_image = filters.apply_filter_1b(original_image)

plt.imshow(new_image, cmap='gray')
plt.show()
imageio.imwrite(f'output_data/{filter_to_apply}-{image_pattern}-{img_name}', new_image)
