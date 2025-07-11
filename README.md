# BEP
Using Segment Anything for the analysis of grains, retrieving GSD's from images. In this file the necessary steps for getting a single GSD from an image of grains are elaborated. 

## Installation SAM
This is the installation for the original Segment Anything code, in this project the Mobile SAM version was used for faster runtime. The installation of Mobile SAM is quite similar but is still mentioned later on. 
The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

Install Segment Anything:

```
pip install git+https://github.com/facebookresearch/segment-anything.git
```


The following optional dependencies are necessary for mask post-processing, saving masks in COCO format, the example notebooks. `jupyter` is also required to run the example notebooks.

```
pip install opencv-python pycocotools matplotlib 
```

First download a model checkpoint. 
### <a name="Models"></a>Model Checkpoints

Three model versions of the model are available with different backbone sizes. These models can be instantiated by running

```
from segment_anything import sam_model_registry
sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
```

Click the links below to download the checkpoint for the corresponding model type.

- **`default` or `vit_h`: [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)** (Biggest dataset, best results but also longest runtime)
- `vit_l`: [ViT-L SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth) (Smaller dataset than vit_h but bigger than base)
- `vit_b`: [ViT-B SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) (Base dataset, fastest runtime but worse results)

## Installation Mobile SAM

The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

Install Mobile Segment Anything:

```
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
```

### Model checkpoint
Download the model weights from the [checkpoints](https://drive.google.com/file/d/1dE-YAG-1mFCBmao2rHDp0n-PP4eH7SjE/view?usp=sharing).

From now on the Mobile SAM model will be used in the examples

## Finding Card for scaling
Finding the card for scaling can be done using the predictor class from segment anything where input coordinates are needed. For automated detection CLIP can be used, for some images manual detection is still needed. Depending on the preferation certain packages need to be installed.
for automatic:
```
pip install Pillow
pip install git+https://github.com/openai/CLIP.git
```

## Set-up
Importing packages. 
Here SAM and Mobile SAM are imported as I used the SamPredictor class, but the same can be done with the Mobile Sam model, thus importing one of the two is enough. 
```
import torch
import numpy as np
import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from PIL import Image
import clip
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import os
from os import listdir
import matplotlib.pyplot as plt
```


### Loading models
Note that loading the clip model is only necessary for automatic scaling.
```
model_type = "vit_t"
sam_checkpoint = "weight/mobile_sam.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"

mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam.to(device=device)
mobile_sam.eval()

clip_model, preprocess = clip.load("ViT-B/32", device=device)
```

### Defining functions
- The show_anns function is from the Segment Anything team, and is used for showing the generated masks with random colours. The colours can be changed.
- The resize function makes sure the input image does not exceed a certain size to make the runtime faster.
- The duplicates function finds masks that are overlapping partially, the threshold can be changed. It returns an array with the doublemasks.
- After all the masks are found the fit_ellipse function fits ellipses on the binary masks and returns the ellipses with center, major and minor axes and the angle.
```
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def resize(image, max_side=1024):
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h))

def duplicates(masks, min_iou = 0.85):
    n = len(masks)
    dupes = []
    for i in range(n):
        for j in range(i + 1, n):
            mask1 = masks[i]['segmentation']
            mask2 = masks[j]['segmentation']
            intersec = np.logical_and(mask1, mask2).sum()
            union = np.logical_or(mask1, mask2).sum()
            if union == 0:
                return 0.0
            iou = intersec / union
            if iou > min_iou:
                dupes.append((i, j, iou))
    return dupes

def fit_ellipse(mask):
    mask_uint8 = mask.astype(np.uint8) * 255 if mask.dtype == bool else mask
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea)
    ellipse = cv2.fitEllipse(largest)
    return ellipse
```

### Adding image(s)
I got two folders of images that needed to be analyzed sent to me. One had photos taken at the inner bends of rivers and one at the outer bends. Here code lines are written so that these images were more easy accesible.
```
folder = 'C:/Users/Luc Velders/OneDrive - Delft University of Technology/BEP/LUC_Outer_Bend'
imagelist = []
for images in os.listdir(folder):
    imagelist.append(images)
file = 'LUC_Outer_Bend/'
OuterBend = [file + x for x in imagelist]

folder1 = 'C:/Users/Luc Velders/OneDrive - Delft University of Technology/BEP/LUC_Inner Bend'
imagelist1 = []
for images in os.listdir(folder1):
    imagelist1.append(images)
file1 = 'LUC_Inner Bend/'
InnerBend = [file1 + x for x in imagelist1]
```
```
image = cv2.imread(OuterBend[2])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = resize(image)
```
### Automatic Mask Generation
```
mask_generator = SamAutomaticMaskGenerator(
    model = mobile_sam,
    points_per_side = 32,
    points_per_batch = 64,
    pred_iou_thresh=0.92,
    stability_score_thresh=0.95,
    stability_score_offset=1.0,
    box_nms_thresh=0.85,
    crop_n_layers=0,
    crop_nms_thresh=0.85,
    crop_overlap_ratio=0.341,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=150,
    output_mode="binary_mask"
)
```
```
masks = mask_generator.generate(image)
```
Plotting figure with the generated masks:
```
plt.figure(figsize=(10,10))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.title('Segmented Masks')
```

### Automatic scaling
```
def find_card(image, masks, text_prompt="card", device='cuda' if torch.cuda.is_available() else 'cpu'):
    pil_image = Image.fromarray(image)
    text_tokens = clip.tokenize([text_prompt]).to(device)
    best_score = -1
    best_mask = None

    for mask in masks:
        seg = mask["segmentation"]
        x0, y0, w, h = cv2.boundingRect(seg.astype(np.uint8))
        if w == 0 or h == 0:
            continue

        crop = pil_image.crop((x0, y0, x0 + w, y0 + h))
        crop_tensor = preprocess(crop).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = clip_model.encode_image(crop_tensor)
            text_features = clip_model.encode_text(text_tokens)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).item()

        if similarity > best_score:
            best_score = similarity
            best_mask = seg.astype(np.uint8)

    return best_mask, best_score
```
CLIP is used to find the mask belonging to the card and its area.
```
cardmask, cardscore = find_card(image, masks, text_prompt="card")
```
### Getting further information
Finding the duplicate and overlapping masks and removing them. Fitting the ellipses on the masks. And using the scale to find the b-axes in mm. By the b-axis, a line was added where a threshold can be put in to remove very large or very small masks, this is checked with a b-axis not being larger that for example 2/3 of one side of the image. 
```
dupes = duplicates(masks, min_iou = 0.25)
doublemasks = []
for i in dupes:
    area1 = masks[i[0]]['area']
    area2 = masks[i[1]]['area']
    if area1 < area2:
        doublemasks.append(i[0])
    if area1 > area2:
        doublemasks.append(i[1])
doublemasks = list(set(doublemasks))
doublemasks.sort(reverse=True)
for i in doublemasks:
    masks.pop(i)

ellipses = []
for i, a in enumerate(masks):
    mask = a['segmentation']
    ellipse = fit_ellipse(mask)
    ellipses.append({"index": i, "center": ellipse[0], "axes": ellipse[1], "angle": ellipse[2], "area": a["area"]})

b = image.copy()
for i in ellipses:
    cv2.ellipse(b, (tuple(map(int, i["center"])), tuple(map(int, i["axes"])), i["angle"]), (0, 255, 0), 2)

card_area = cardmask.sum()
scale = ((85.60 * 53.98) - 4 * (3.18**2) + np.pi * 3.18**2) / card_area # mm^2 / pixel

pix_b_axis = []
for i in range(len(ellipses)):
    pix_b_axis.append(max(ellipses[i]['axes'])) #ellipses[i]['index']])
        
index = []
for i in range(len(pix_b_axis)):
    if (pix_b_axis[i]) >= ((2 / 3) * max(np.shape(image))):
        index.append(i)
index.sort(reverse=True)
for i in index:
    pix_b_axis.pop(i)
    
b_axis = [x * np.sqrt(scale) for x in pix_b_axis]
b_axis.sort()
```

### Plotting the information
```
plt.figure(figsize=(10,10))
plt.subplot(221)
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.title('Segmented Masks')

plt.subplot(222)
plt.imshow(b)
plt.axis('off')
plt.title("Fitted ellipses")

plt.subplot(223)
plt.imshow(cardmask, cmap='gray')
plt.title('Found Card Mask')

cdf = np.linspace(0, 1, len(b_axis))
ax = plt.subplot(224)
plt.plot(b_axis, cdf, linestyle='-')
plt.xscale('log')
plt.xlabel('b-axis (mm)')
plt.ylabel('CDF')
plt.title('Grain Size Distribution - b-axis')
plt.grid(True)
ax.set_aspect('equal')
plt.show()
plt.tight_layout();
```

### Finding card manually
The mask of the card can also be found manually, and sometimes has to be found manually as the automatic has trouble finding the card for some images. 1 coordinate located on the card is needed, the predictor then finds three masks. Using the best of the three masks and calculating the sum of the mask gives the area of the card in pixels.
```
predictor = SamPredictor(mobile_sam)
predictor.set_image(image)

input_point = np.array([[500, 500]])
input_label = np.array([1])

masks, scores, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    

for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show() 
```
```
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```
```
@article{mobile_sam,
  title={Faster Segment Anything: Towards Lightweight SAM for Mobile Applications},
  author={Zhang, Chaoning and Han, Dongshen and Qiao, Yu and Kim, Jung Uk and Bae, Sung-Ho and Lee, Seungkyu and Hong, Choong Seon},
  journal={arXiv preprint arXiv:2306.14289},
  year={2023}
}
```
