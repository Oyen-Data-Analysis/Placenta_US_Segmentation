import numpy as np
import torch
from torch.nn.functional import threshold, normalize

# Insert the model
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide

# Load the model
# you can download other checkpoints of pretrained SAM models:
# Vit-h(default, and largest size), Vit-l(medium size), and Vit-B(smallest size)
# sam_checkpoint = "sam_vit_h_4b8939.pth"
sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"

device = 'cuda:1'

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# Add prompts
# single prompt (box)
input_box = np.array([[X, Y, X, Y]])  # XYXY represent the four corners of the box
input_box = ResizeLongestSide(sam.image_encoder.img_size).apply_boxes(input_box, (
x, y))  # x, y is the height and width of the input image
input_box_torch = torch.as_tensor(input_box, dtype=torch.float, device=device)
input_box_torch = input_box_torch[None, :]

# multiple prompts (points)
input_points = np.array([[X, Y], [X, Y]])  # the point coordinate
input_label = np.array([1, 1])  # you can choose either 1 (foreground point) or 0 (background point)
point_coords = ResizeLongestSide(sam.image_encoder.img_size).apply_coords(input_points, (x, y))
coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=device)
labels_torch = torch.as_tensor(input_label, dtype=torch.int, device=device)
coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
input_points_torch = (coords_torch, labels_torch)

# multiple prompts (boxes)
input_boxes = torch.tensor([
    [X1, Y1, X1, Y1],
    [X2, Y2, X2, Y2],
    [X3, Y3, X3, Y3],
    [X4, Y4, X4, Y4]
], device=device)

input_boxes_pre = ResizeLongestSide(sam.image_encoder.img_size).apply_boxes_torch(input_boxes, (x, y))

# Forward propagation of the SAM model
# Each time, only a single image can be processed.
# The pixel values of input_image should be normalized to [0, 1] for now
input_img = input_image.to(device)  # input image in 3D or 1D form with shape [C, H, W]
input_img_resize = torch.nn.functional.interpolate(input_img.unsqueeze(0), size=(1024, 1024), mode='bilinear',
                                                   align_corners=False).squeeze(0)  # Resize the image to (1024, 1024)
input_img_resize = input_img_resize * 255  # The pixels value of the image should be in [0, 255]
input_img_resize = input_img_resize[None, :, :, :].contiguous()
size_input_img = tuple(input_img.shape[1:])
size_input_img_resize = tuple(input_img_resize.shape[-2:])
# SAM model
input_img_resize = sam.preprocess(input_img_resize)

image_features = sam.image_encoder(input_img_resize)

sparse_embeddings, dense_embeddings = sam.prompt_encoder(
    points=None,  # replace None with input_points_torch
    boxes=input_boxes_pre,  # you can use either single prompt or multiple prompts here
    masks=None,
)

low_res_masks, iou_predictions = sam.mask_decoder(
    image_embeddings=image_features,
    image_pe=sam.prompt_encoder.get_dense_pe(),
    sparse_prompt_embeddings=sparse_embeddings,
    dense_prompt_embeddings=dense_embeddings,
    multimask_output=False,
)

masks_ori = sam.postprocess_masks(low_res_masks, size_input_img_resize, size_input_img)

masks_ori = normalize(threshold(masks_ori, 0.0,
                                0))  # this should generates binary masks with shape [N, 1, H, W], N is the number of masks, H and W is the original size of the input image
# masks_ori = torch.sigmoid(masks_ori) # you can generate probability masks if you want