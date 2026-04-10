'''
usage: python gen_diff.py -h
'''

from __future__ import print_function

import argparse

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Input
import imageio

from tensorflow.keras.models import load_model
from configs import bcolors
from utils import *
import matplotlib.pyplot as plt

CLASS_NAMES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

def save_with_info(img, label1, label2, idx, true_label=None, cov_before=None, cov_after=None):
    plt.figure(figsize=(4, 5))

    # 이미지 표시
    plt.imshow(img)
    plt.axis('off')

    # 텍스트 구성
    text = f"Index: {idx}\n"
    
    if true_label is not None:
        text += f"True: {true_label}\n"

    text += f"Preds: [{CLASS_NAMES[label1]}, {CLASS_NAMES[label2]}]\n"

    if cov_before is not None and cov_after is not None:
        text += f"Coverage: {cov_before:.4f} → {cov_after:.4f}"

    # 텍스트 추가
    plt.title(text, fontsize=10)

    # 저장
    plt.savefig(f"./results/result_{idx}_{label1}_{label2}.png")
    plt.close()

# read the parameter
# argument parsing
parser = argparse.ArgumentParser(description='Main function for difference-inducing input generation in CIFAR10 dataset')
parser.add_argument('transformation', help="realistic transformation type", choices=['light', 'occl', 'blackout'])
parser.add_argument('weight_diff', help="weight hyperparm to control differential behavior", type=float)
parser.add_argument('weight_nc', help="weight hyperparm to control neuron coverage", type=float)
parser.add_argument('step', help="step size of gradient descent", type=float)
parser.add_argument('seeds', help="number of seeds of input", type=int)
parser.add_argument('grad_iterations', help="number of iterations of gradient descent", type=int)
parser.add_argument('threshold', help="threshold for determining neuron activated", type=float)
parser.add_argument('-t', '--target_model', help="target model that we want it predicts differently",
                    choices=[0, 1, 2], default=0, type=int)
parser.add_argument('-sp', '--start_point', help="occlusion upper left corner coordinate", default=(0, 0), type=tuple)
parser.add_argument('-occl_size', '--occlusion_size', help="occlusion size", default=(10, 10), type=tuple)

args = parser.parse_args()

# input image dimensions
img_rows, img_cols = 32, 32
# the data, shuffled and split between train and test sets
(_, _), (x_test, y_test) = cifar10.load_data()


input_shape = (img_rows, img_cols, 3)

x_test = x_test.astype('float32')
x_test /= 255

# define input tensor as a placeholder
input_tensor = Input(shape=input_shape)

# load multiple models sharing same input tensor
model1 = load_model("model1.h5")
model2 = load_model("model2.h5")

# init coverage table
model_layer_dict1, model_layer_dict2 = init_coverage_tables(model1, model2)

# ==============================================================================================
# start gen inputs
for sample_idx  in range(args.seeds):
    rand_idx = random.randrange(len(x_test))
    gen_img = np.expand_dims(x_test[rand_idx], axis=0)
    orig_img = gen_img.copy()
    # first check if input already induces differences
    pred1 = model1.predict(gen_img)
    pred2 = model2.predict(gen_img)

    label1 = np.argmax(pred1)
    label2 = np.argmax(pred2)


    if not label1 == label2:
        print(bcolors.OKGREEN + 'input already causes different outputs: {}, {}'.format(label1, label2
                                                                                            ) + bcolors.ENDC)

        update_coverage(gen_img, model1, model_layer_dict1, args.threshold)
        update_coverage(gen_img, model2, model_layer_dict2, args.threshold)


        print(bcolors.OKGREEN + 'covered neurons percentage %d neurons %.3f, %d neurons %.3f'
      % (len(model_layer_dict1), neuron_covered(model_layer_dict1)[2],
         len(model_layer_dict2), neuron_covered(model_layer_dict2)[2])
      + bcolors.ENDC)
        averaged_nc = (
    neuron_covered(model_layer_dict1)[0] +
    neuron_covered(model_layer_dict2)[0]
) / float(
    neuron_covered(model_layer_dict1)[1] +
    neuron_covered(model_layer_dict2)[1]
)
        print(bcolors.OKGREEN + 'averaged covered neurons %.3f' % averaged_nc + bcolors.ENDC)

        gen_img_deprocessed = deprocess_image(gen_img)

        # save the result to disk
        save_with_info(
    gen_img_deprocessed,
    label1,
    label2,
    sample_idx,
    true_label=CLASS_NAMES[y_test[rand_idx][0]],
    cov_before=None,
    cov_after=averaged_nc
)
        continue

    # if all label agrees
    orig_label = label1
    layer_name1, index1 = neuron_to_cover(model_layer_dict1)
    layer_name2, index2 = neuron_to_cover(model_layer_dict2)

    # construct joint loss function
    if args.target_model == 0:
        loss1 = -args.weight_diff * K.mean(model1.output[..., orig_label])
        loss2 = K.mean(model2.output[..., orig_label])

    elif args.target_model == 1:
        loss1 = K.mean(model1.output[..., orig_label])
        loss2 = -args.weight_diff * K.mean(model2.output[..., orig_label])

    elif args.target_model == 2:
        loss1 = K.mean(model1.output[..., orig_label])
        loss2 = K.mean(model2.output[..., orig_label])

    loss1_neuron = K.mean(model1.get_layer(layer_name1).output[..., index1])
    loss2_neuron = K.mean(model2.get_layer(layer_name2).output[..., index2])

    layer_output = (loss1 + loss2 ) + args.weight_nc * (loss1_neuron + loss2_neuron)

    # for adversarial image generation
    final_loss = K.mean(layer_output)

    