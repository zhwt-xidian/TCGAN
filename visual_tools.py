import matplotlib.pyplot as plt
import numpy as np
import io
import torchvision.transforms as transforms
from PIL import Image
import torch

def plot_two_subplots(x, y1, y2, y3, title1='Subplot 1', title2='Subplot 2', title3='Subplot3', label1=None, label2=None, label3=None):

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 6))
    ax1.plot(x, y1, color='blue', label=label1)
    ax1.set_title(title1)
    ax1.legend()

    ax2.plot(x, y2, color='red', label=label2)
    ax2.set_title(title2)
    ax2.legend()

    ax3.plot(x, y3, color='green', label=label3)
    ax3.set_title(title3)
    ax3.legend()

    plt.subplots_adjust(hspace=0.5)

def plt_to_tensor(plt):
    """
    Transform Matplotlib picture to Tensor
    """
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    image_array = np.array(image)
    image_tensor = torch.from_numpy(image_array)
    image_tensor.transpose(1, 2, 0)
    return image_tensor