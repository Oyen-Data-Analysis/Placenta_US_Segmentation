import torch
from typing import Dict, Optional, Sequence, Tuple, Union
import numpy as np
import tifffile as tiff
def to_tiff(x, path, is_normalized=True):
  try:
    x = np.squeeze(x)
  except:
    pass

  try:
    x = torch.squeeze(x).numpy()
  except:
    pass

  print(x.shape, path)

  if len(x.shape) == 3:
    # n_slice, n_x, n_y = x.shape
    n_x, n_y, n_slice = x.shape

    if is_normalized:
      for i in range(n_slice):
        x[:, :, i] -= np.amin(x[:, :, i])
        x[:, :, i] /= np.amax(x[:, :, i])
        #
        x[:, :, i] *= 255

      x = x.astype(np.uint8)
      # x = x.astype(np.float32)
  x = x.astype(np.float32)
  tiff.imwrite(path, x, imagej=True, ijmetadata={'Slice': n_slice})


def center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
  """
  Apply a center crop to the input real image or batch of real images.

  Args:
      data: The input tensor to be center cropped. It should
          have at least 2 dimensions and the cropping is applied along the
          last two dimensions.
      shape: The output shape. The shape should be smaller
          than the corresponding dimensions of data.

  Returns:
      The center cropped image.
  """
  if not (0 < shape[0] <= data.shape[-2] and 0 < shape[1] <= data.shape[-1]):
    raise ValueError("Invalid shapes.")

  w_from = (data.shape[-2] - shape[0]) // 2
  h_from = (data.shape[-1] - shape[1]) // 2
  w_to = w_from + shape[0]
  h_to = h_from + shape[1]

  return data[..., w_from:w_to, h_from:h_to]


def normlize(data):
  """
  0-1 normlization
  Args:
      data: The input tensor
  Returns:
      The 0-1 normlized data.
  """
  return (data - data.min()) / (data.max() - data.min())