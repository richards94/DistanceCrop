from torchvision.transforms import RandomResizedCrop
import torch
import random
import numpy as np
import math
from torch.distributions.uniform import Uniform
import torchvision.transforms.functional as F


class DistanceCrop(RandomResizedCrop):
    def __init__(self, center_dist_ratio=0.2, **kwargs):
        super().__init__(**kwargs)
        self.center_dist_ratio = center_dist_ratio
        self.uniform = Uniform(0,1)

    def get_params(self, img, box, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        # width, height = F._get_image_size(img)
        width, height = img.size
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                h0, w0, h1, w1 = box
                ch0 = min(max(int(height * h0) - h//2, 0), height - h)
                ch1 = min(max(int(height * h1) - h//2, 0), height - h)
                cw0 = min(max(int(width * w0) - w//2, 0), width - w)
                cw1 = min(max(int(width * w1) - w//2, 0), width - w)

                i = ch0 + int((ch1 - ch0) * self.uniform.sample())
                j = cw0 + int((cw1 - cw0) * self.uniform.sample())
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w
    
    def center_distance(self, img, box, i1, j1, h1, w1, i2, j2, h2, w2):
        width, height = img.size
        box_h0, box_w0, box_h1, box_w1 = box
        box_width = width * (box_w1 - box_w0)
        box_height = height * (box_h1 - box_h0)
        box_length = math.sqrt(box_width*box_width + box_height*box_height)
        crop1_centerh = i1 + h1 // 2
        crop1_centerw = j1 + w1 // 2
        crop2_centerh = i2 + h2 // 2
        crop2_centerw = j2 + w2 // 2
        crop_center_length = math.sqrt((crop1_centerh-crop2_centerh)**2 + 
                                       (crop1_centerw-crop2_centerw)**2)
        cur_center_dist_ratio = crop_center_length / box_length
        if cur_center_dist_ratio >= self.center_dist_ratio:
            return True
        else:
            return False

    def forward(self, img, box):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        is_ok = False
        for _ in range(10):
            i1, j1, h1, w1 = self.get_params(img, box, self.scale, self.ratio)
            i2, j2, h2, w2 = self.get_params(img, box, self.scale, self.ratio)
            is_ok = self.center_distance(img, box, i1, j1, h1, w1, i2, j2, h2, w2)
            if is_ok:
                break
        if not is_ok:
            i1, j1, h1, w1 = self.get_params(img, box, self.scale, self.ratio)
            i2, j2, h2, w2 = self.get_params(img, box, self.scale, self.ratio)
        
        view1 = F.resized_crop(img, i1, j1, h1, w1, self.size, self.interpolation)
        view2 = F.resized_crop(img, i2, j2, h2, w2, self.size, self.interpolation)
        
        return view1, view2
        
