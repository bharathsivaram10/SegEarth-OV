from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from segearth_segmentor import SegEarthSegmentation
import os
import torch
import numpy as np
from matplotlib.patches import Patch
import time

name_list = ['background', 'tree', 'grass', 'road', 'building', 'pavement', 'vehicle', 'farmland']  # example classes

IMG_DIR = '/home/bharath/aerial_imgs'
OUTPUT_DIR = './seg_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)  # create output folder if it doesn't exist

# Define a library of distinct colors (RGB tuples)
COLOR_LIBRARY = [
    (0, 0, 0),        # black
    (255, 0, 0),      # red
    (0, 255, 0),      # green
    (0, 0, 255),      # blue
    (255, 255, 0),    # yellow
    (255, 165, 0),    # orange
    (128, 0, 128),    # purple
    (0, 255, 255),    # cyan
    (255, 192, 203),  # pink
    (128, 128, 128),  # gray
]

if len(name_list) > len(COLOR_LIBRARY):
    raise ValueError(f"Number of classes ({len(name_list)}) exceeds number of available colors ({len(COLOR_LIBRARY)}).")

# Randomly assign a color to each class
np.random.seed(42)  # for reproducibility
class_colors = {name: COLOR_LIBRARY[i] for i, name in enumerate(name_list)}


def load_model():
    with open('./configs/my_name.txt', 'w') as f:
        f.write("\n".join(name_list))

    model = SegEarthSegmentation(
        clip_type='CLIP',
        vit_type='ViT-B/16',
        model_type='SegEarth',
        ignore_residual=True,
        feature_up=True,
        feature_up_cfg=dict(
            model_name='jbu_one',
            model_path='simfeatup_dev/weights/xclip_jbu_one_million_aid.ckpt'),
        cls_token_lambda=-0.3,
        name_path='./configs/my_name.txt',
        prob_thd=0.1,
    )
    return model


def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                             [0.26862954, 0.26130258, 0.27577711])
    ])
    img_tensor = transform(img).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
    return img, img_tensor


def apply_colors(seg_pred):
    """Convert class indices to RGB colors using class_colors dict"""
    h, w = seg_pred.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, class_name in enumerate(name_list):
        color_img[seg_pred == idx] = class_colors[class_name]
    return color_img


def main():
    model = load_model()

    # get all image paths from IMG_DIR
    img_files = [os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]

    for img_path in img_files:
        img, img_tensor = preprocess_image(img_path)

        # run prediction
        t1 = time.time()
        seg_pred = model.predict(img_tensor, data_samples=None)
        t2 = time.time()
        print(f'Inference time: {t2-t1}')
        seg_pred = seg_pred.data.cpu().numpy().squeeze(0).astype(int)

        # apply colors
        seg_color = apply_colors(seg_pred)

        # save visualization with legend
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        ax[0].imshow(img)
        ax[0].axis('off')
        ax[0].set_title("Original Image")

        ax[1].imshow(seg_color)
        ax[1].axis('off')
        ax[1].set_title("Segmentation")

        # create legend patches
        legend_elements = [Patch(facecolor=np.array(color)/255.0, label=name)
                           for name, color in class_colors.items()]
        ax[1].legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        plt.tight_layout()

        base_name = os.path.basename(img_path)
        save_path = os.path.join(OUTPUT_DIR, f"seg_{base_name}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()