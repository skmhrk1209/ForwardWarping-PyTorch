import torch
from torch import nn
import numpy as np
from skimage import io
import argparse
import warping
import pfm
import os


def main(args):

    left_image = io.imread(os.path.join(args.dirname, "im0.png")).astype(np.float32) / ((1 << 8) -1)
    right_image = io.imread(os.path.join(args.dirname, "im1.png")).astype(np.float32) / ((1 << 8) -1)
    left_disparity = pfm.readPFM(os.path.join(args.dirname, "disp0GT.pfm"))[0].astype(np.float32)[..., np.newaxis]
    right_disparity = pfm.readPFM(os.path.join(args.dirname, "disp1GT.pfm"))[0].astype(np.float32)[..., np.newaxis]
    left_occlusion = io.imread(os.path.join(args.dirname, "mask0nocc.png")).astype(np.float32)[..., np.newaxis] / ((1 << 8) -1)
    right_occlusion = io.imread(os.path.join(args.dirname, "mask1nocc.png")).astype(np.float32)[..., np.newaxis] / ((1 << 8) -1)

    left_image = torch.from_numpy(left_image).permute(2, 0, 1).unsqueeze(0)
    right_image = torch.from_numpy(right_image).permute(2, 0, 1).unsqueeze(0)
    left_disparity = torch.from_numpy(left_disparity).permute(2, 0, 1).unsqueeze(0)
    right_disparity = torch.from_numpy(right_disparity).permute(2, 0, 1).unsqueeze(0)
    left_occlusion = torch.from_numpy(left_occlusion).permute(2, 0, 1).unsqueeze(0)
    right_occlusion = torch.from_numpy(right_occlusion).permute(2, 0, 1).unsqueeze(0)
    
    left_disparity = torch.where(torch.isfinite(left_disparity), left_disparity, torch.zeros_like(left_disparity))
    right_disparity = torch.where(torch.isfinite(right_disparity), right_disparity, torch.zeros_like(right_disparity))

    left_occlusion = torch.where(left_occlusion < 1, torch.zeros_like(left_occlusion), left_occlusion)
    right_occlusion = torch.where(right_occlusion < 1, torch.zeros_like(right_occlusion), right_occlusion)

    backward_left_image = warping.warp_backward(right_image, left_disparity, invert=True) * left_occlusion
    backward_right_image = warping.warp_backward(left_image, right_disparity, invert=False) * right_occlusion

    forward_left_image = warping.warp_forward(right_image, right_disparity, right_occlusion, invert=False)
    forward_right_image = warping.warp_forward(left_image, left_disparity, left_occlusion, invert=True)

    backward_left_image = backward_left_image.squeeze(0).permute(1, 2, 0).numpy()
    backward_right_image = backward_right_image.squeeze(0).permute(1, 2, 0).numpy()

    forward_left_image = forward_left_image.squeeze(0).permute(1, 2, 0).numpy()
    forward_right_image = forward_right_image.squeeze(0).permute(1, 2, 0).numpy()

    io.imsave("backward0.png", backward_left_image)
    io.imsave("backward1.png", backward_right_image)
    io.imsave("forward0.png", forward_left_image)
    io.imsave("forward1.png", forward_right_image)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Forward warping example")
    parser.add_argument("dirname", type=str)
    args = parser.parse_args()

    main(args)
