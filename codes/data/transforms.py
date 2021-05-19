import cv2
import random
import numpy as np
import torchvision.transforms.functional as F
import torchvision.transforms as transforms

def mod_crop(img, scale):
    """Mod crop images, used during testing.

    Args:
        img (ndarray): Input image.
        scale (int): Scale factor.

    Returns:
        ndarray: Result image.
    """
    img = img.copy()
    if img.ndim in (2, 3):
        h, w = img.shape[0], img.shape[1]
        h_remainder, w_remainder = h % scale, w % scale
        img = img[:h - h_remainder, :w - w_remainder, ...]
    else:
        raise ValueError(f'Wrong img ndim: {img.ndim}.')
    return img

def paired_random_crop(img_gts, img_lqs, if_fix, patch_size, gt_path):

    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    h_lq, w_lq, _ = img_lqs[0].shape
    h_gt, w_gt, _ = img_gts[0].shape

    if h_gt != h_lq or w_gt != w_lq:
        raise ValueError(
            f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {1}x ',
            f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < patch_size or w_lq < patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({patch_size}, {patch_size}). '
                         f'Please remove {gt_path}.')

    # randomly choose top and left coordinates for lq patch
    if if_fix:
        top = random.randint(0, h_lq - patch_size)
        left = random.randint(0, w_lq - patch_size)
        img_lqs = [v[top:top + patch_size, left:left + patch_size, ...] for v in img_lqs]
        img_gts = [v[top:top + patch_size, left:left + patch_size, ...] for v in img_gts]
    else:
        ratio_h = np.random.uniform(0.6, 1.0)
        ratio_w = np.random.uniform(0.6, 1.0)
        size_h = round(h_lq * ratio_h)
        size_w = round(w_lq * ratio_w)
        top = random.randint(0, h_lq - size_h)
        left = random.randint(0, w_lq - size_w)
        img_lqs = [v[top:top + size_h, left:left + size_w, ...] for v in img_lqs]
        img_gts = [v[top:top + size_h, left:left + size_w, ...] for v in img_gts]

    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    return img_gts, img_lqs

def unpaired_random_crop(img_lqs, img_refs, if_fix, patch_size):

    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]
    if not isinstance(img_refs, list):
        img_refs = [img_refs]

    h_lq, w_lq, _ = img_lqs[0].shape
    h_ref, w_ref, _ = img_refs[0].shape

    if h_lq < patch_size or w_lq < patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({patch_size}, {patch_size}). ')

    # randomly choose top and left coordinates for lq patch
    if if_fix:
        top = random.randint(0, h_lq - patch_size)
        left = random.randint(0, w_lq - patch_size)
        img_lqs = [v[top:top + patch_size, left:left + patch_size, ...] for v in img_lqs]
        top = random.randint(0, h_ref - patch_size)
        left = random.randint(0, w_ref - patch_size)
        img_refs = [v[top:top + patch_size, left:left + patch_size, ...] for v in img_refs]
    else:
        ratio_h = np.random.uniform(0.6, 1.0)
        ratio_w = np.random.uniform(0.6, 1.0)
        size_h = round(min(h_lq, h_ref) * ratio_h)
        size_w = round(min(w_lq, w_ref) * ratio_w)
        top = random.randint(0, h_lq - size_h)
        left = random.randint(0, w_lq - size_w)
        img_lqs = [v[top:top + size_h, left:left + size_w, ...] for v in img_lqs]
        top = random.randint(0, h_ref - size_h)
        left = random.randint(0, w_ref - size_w)
        img_refs = [v[top:top + size_h, left:left + size_w, ...] for v in img_refs]

    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    if len(img_refs) == 1:
        img_refs = img_refs[0]
    return img_lqs, img_refs

def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees) OR brightness OR saturation.

    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        flows (list[ndarray]: Flows to be augmented. If the input is an
            ndarray, it will be transformed to a list.
            Dimension is (h, w, 2). Default: None.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.

    """
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            cv2.flip(img, 1, img)
        if vflip:  # vertical
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:  # horizontal
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:  # vertical
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        if return_status:
            return imgs, (hflip, vflip, rot90)
        else:
            return imgs


def img_rotate(img, angle, center=None, scale=1.0):
    """Rotate image.

    Args:
        img (ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees. Positive values mean
            counter-clockwise rotation.
        center (tuple[int]): Rotation center. If the center is None,
            initialize it as the center of the image. Default: None.
        scale (float): Isotropic scale factor. Default: 1.0.
    """
    (h, w) = img.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_img = cv2.warpAffine(img, matrix, (w, h))
    return rotated_img