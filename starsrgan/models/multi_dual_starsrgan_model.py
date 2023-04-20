import math
import torch
import random
import numpy as np
from collections import OrderedDict
from torch.nn import functional as F

from starsrgan.utils.registry import MODEL_REGISTRY
from starsrgan.models.stargan_model import StarGANModel
from starsrgan.utils.img import DiffJPEG, USMSharp, filter2D
from starsrgan.utils.transform import paired_random_crop_return_indexes

from starsrgan.utils.degradation import (
    random_add_gaussian_noise_pt_info,
    random_add_poisson_noise_pt_info,
)
from starsrgan.utils.degradation import (
    only_generate_gaussian_noise_pt_info,
    only_generate_poisson_noise_pt_info,
)


@MODEL_REGISTRY.register()
class MultiDualStarSRGANModel(StarGANModel):
    """StarSRGAN Model for StarSRGAN.

    It mainly performs:
    1. randomly synthesize LQ images in GPU tensors
    2. optimize the networks with GAN training.
    """

    def __init__(self, opt):
        super(MultiDualStarSRGANModel, self).__init__(opt)
        self.jpeger = DiffJPEG(
            differentiable=False
        ).cuda()  # simulate JPEG compression artifacts
        self.usm_sharpener = USMSharp().cuda()  # do usm sharpening
        self.queue_size = opt.get("queue_size", 180)
        self.resize_mode_list = ["area", "bilinear", "bicubic"]
        self.opt_train = opt["datasets"]["train"]
        num_degradation_params = 4 * 2 + 2  # kernel
        num_degradation_params += 4 * 2  # resize
        num_degradation_params += 4 * 2  # noise
        num_degradation_params += 3 + 2 + 2  # jpeg
        self.num_degradation_params = num_degradation_params
        self.road_map = [0, 10, 10 + 8, 10 + 8 + 8, 10 + 8 + 8 + 7]
        # [0, 10, 18, 26, 33]

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, "queue_lr"):
            assert (
                self.queue_size % b == 0
            ), f"queue size {self.queue_size} should be divisible by batch size {b}"
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            self.queue_lr[
                self.queue_ptr : self.queue_ptr + b, :, :, :
            ] = self.lq.clone()
            self.queue_gt[
                self.queue_ptr : self.queue_ptr + b, :, :, :
            ] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b

    @torch.no_grad()
    def feed_data(self, data_all):
        """Accept data from dataloader, and then add two-order degradations to obtain LQ images."""
        if self.is_train:
            # training data synthesis
            self.degradation_degree = random.choices(
                self.opt["degree_list"], self.opt["degree_prob"]
            )[0]
            data = data_all[self.degradation_degree]

            self.gt = data["gt"].to(self.device)
            self.gt_usm = self.usm_sharpener(self.gt)

            if self.degradation_degree == "severe_degrade_two_stage":
                self.degradation_params = torch.zeros(
                    self.opt_train["batch_size_per_gpu"], self.num_degradation_params
                )

                self.kernel1 = data["kernel1"]["kernel"].to(self.device)
                self.kernel2 = data["kernel2"]["kernel"].to(self.device)
                self.sinc_kernel = data["sinc_kernel"]["kernel"].to(self.device)

                kernel_size_range1 = [
                    self.opt_train["blur_kernel_size_minimum"],
                    self.opt_train["blur_kernel_size"],
                ]
                kernel_size_range2 = [
                    self.opt_train["blur_kernel_size2_minimum"],
                    self.opt_train["blur_kernel_size2"],
                ]
                rotation_range = [-math.pi, math.pi]
                omega_c_range = [np.pi / 3, np.pi]

                self.degradation_params[:, self.road_map[0] : self.road_map[0] + 1] = (
                    data["kernel1"]["kernel_size"].unsqueeze(1) - kernel_size_range1[0]
                ) / (kernel_size_range1[1] - kernel_size_range1[0])
                self.degradation_params[
                    :, self.road_map[0] + 4 : self.road_map[0] + 5
                ] = (
                    data["kernel2"]["kernel_size"].unsqueeze(1) - kernel_size_range2[0]
                ) / (
                    kernel_size_range2[1] - kernel_size_range2[0]
                )
                self.degradation_params[
                    :, self.road_map[0] + 1 : self.road_map[0] + 2
                ] = (
                    data["kernel1"]["sigma_x"].unsqueeze(1)
                    - self.opt_train["blur_sigma"][0]
                ) / (
                    self.opt_train["blur_sigma"][1] - self.opt_train["blur_sigma"][0]
                )
                self.degradation_params[
                    :, self.road_map[0] + 5 : self.road_map[0] + 6
                ] = (
                    data["kernel2"]["sigma_x"].unsqueeze(1)
                    - self.opt_train["blur_sigma2"][0]
                ) / (
                    self.opt_train["blur_sigma2"][1] - self.opt_train["blur_sigma2"][0]
                )
                self.degradation_params[
                    :, self.road_map[0] + 2 : self.road_map[0] + 3
                ] = (
                    data["kernel1"]["sigma_y"].unsqueeze(1)
                    - self.opt_train["blur_sigma"][0]
                ) / (
                    self.opt_train["blur_sigma"][1] - self.opt_train["blur_sigma"][0]
                )
                self.degradation_params[
                    :, self.road_map[0] + 6 : self.road_map[0] + 7
                ] = (
                    data["kernel2"]["sigma_y"].unsqueeze(1)
                    - self.opt_train["blur_sigma2"][0]
                ) / (
                    self.opt_train["blur_sigma2"][1] - self.opt_train["blur_sigma2"][0]
                )
                self.degradation_params[
                    :, self.road_map[0] + 3 : self.road_map[0] + 4
                ] = (data["kernel1"]["rotation"].unsqueeze(1) - rotation_range[0]) / (
                    rotation_range[1] - rotation_range[0]
                )
                self.degradation_params[
                    :, self.road_map[0] + 7 : self.road_map[0] + 8
                ] = (data["kernel2"]["rotation"].unsqueeze(1) - rotation_range[0]) / (
                    rotation_range[1] - rotation_range[0]
                )
                self.degradation_params[
                    :, self.road_map[0] + 8 : self.road_map[0] + 9
                ] = (
                    data["sinc_kernel"]["kernel_size"].unsqueeze(1)
                    - kernel_size_range1[0]
                ) / (
                    kernel_size_range1[1] - kernel_size_range1[0]
                )
                self.degradation_params[:, self.road_map[0] + 9 : self.road_map[1]] = (
                    data["sinc_kernel"]["omega_c"].unsqueeze(1) - omega_c_range[0]
                ) / (omega_c_range[1] - omega_c_range[0])

                ori_h, ori_w = self.gt.size()[2:4]

                # ----------------------- The first degradation process ----------------------- #
                # blur
                out = filter2D(self.gt_usm, self.kernel1)
                # random resize
                updown_type = random.choices(
                    ["up", "down", "keep"], self.opt["resize_prob"]
                )[0]
                if updown_type == "up":
                    scale = np.random.uniform(1, self.opt["resize_range"][1])
                elif updown_type == "down":
                    scale = np.random.uniform(self.opt["resize_range"][0], 1)
                else:
                    scale = 1
                mode = random.choice(self.resize_mode_list)
                out = F.interpolate(out, scale_factor=scale, mode=mode)
                normalized_scale = (scale - self.opt["resize_range"][0]) / (
                    self.opt["resize_range"][1] - self.opt["resize_range"][0]
                )
                onehot_mode = torch.zeros(len(self.resize_mode_list))
                for index, mode_current in enumerate(self.resize_mode_list):
                    if mode_current == mode:
                        onehot_mode[index] = 1
                self.degradation_params[
                    :, self.road_map[1] : self.road_map[1] + 1
                ] = torch.tensor(normalized_scale).expand(self.gt.size(0), 1)
                self.degradation_params[
                    :, self.road_map[1] + 1 : self.road_map[1] + 4
                ] = onehot_mode.expand(self.gt.size(0), len(self.resize_mode_list))
                # add noise
                gray_noise_prob = self.opt["gray_noise_prob"]
                if np.random.uniform() < self.opt["gaussian_noise_prob"]:
                    (
                        sigma,
                        gray_noise,
                        out,
                        self.noise_g_first,
                    ) = random_add_gaussian_noise_pt_info(
                        out,
                        sigma_range=self.opt["noise_range"],
                        clip=True,
                        rounds=False,
                        gray_prob=gray_noise_prob,
                    )
                    normalized_sigma = (sigma - self.opt["noise_range"][0]) / (
                        self.opt["noise_range"][1] - self.opt["noise_range"][0]
                    )
                    self.degradation_params[
                        :, self.road_map[2] : self.road_map[2] + 1
                    ] = normalized_sigma.unsqueeze(1)
                    self.degradation_params[
                        :, self.road_map[2] + 1 : self.road_map[2] + 2
                    ] = gray_noise.unsqueeze(1)
                    self.degradation_params[
                        :, self.road_map[2] + 2 : self.road_map[2] + 4
                    ] = torch.tensor([1, 0]).expand(self.gt.size(0), 2)
                    self.noise_p_first = only_generate_poisson_noise_pt_info(
                        out,
                        scale_range=self.opt["poisson_scale_range"],
                        gray_prob=gray_noise_prob,
                    )
                else:
                    (
                        scale,
                        gray_noise,
                        out,
                        self.noise_p_first,
                    ) = random_add_poisson_noise_pt_info(
                        out,
                        scale_range=self.opt["poisson_scale_range"],
                        gray_prob=gray_noise_prob,
                        clip=True,
                        rounds=False,
                    )
                    normalized_scale = (scale - self.opt["poisson_scale_range"][0]) / (
                        self.opt["poisson_scale_range"][1]
                        - self.opt["poisson_scale_range"][0]
                    )
                    self.degradation_params[
                        :, self.road_map[2] : self.road_map[2] + 1
                    ] = normalized_scale.unsqueeze(1)
                    self.degradation_params[
                        :, self.road_map[2] + 1 : self.road_map[2] + 2
                    ] = gray_noise.unsqueeze(1)
                    self.degradation_params[
                        :, self.road_map[2] + 2 : self.road_map[2] + 4
                    ] = torch.tensor([0, 1]).expand(self.gt.size(0), 2)
                    self.noise_g_first = only_generate_gaussian_noise_pt_info(
                        out,
                        sigma_range=self.opt["noise_range"],
                        gray_prob=gray_noise_prob,
                    )

                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt["jpeg_range"])
                normalized_jpeg_p = (jpeg_p - self.opt["jpeg_range"][0]) / (
                    self.opt["jpeg_range"][1] - self.opt["jpeg_range"][0]
                )
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
                self.degradation_params[
                    :, self.road_map[3] : self.road_map[3] + 1
                ] = normalized_jpeg_p.unsqueeze(1)

                # ----------------------- The second degradation process ----------------------- #
                # blur
                if np.random.uniform() < self.opt["second_blur_prob"]:
                    out = filter2D(out, self.kernel2)
                    self.degradation_params[
                        :, self.road_map[1] - 1 : self.road_map[1]
                    ] = torch.tensor([1]).expand(self.gt.size(0), 1)
                # random resize
                updown_type = random.choices(
                    ["up", "down", "keep"], self.opt["resize_prob2"]
                )[0]
                if updown_type == "up":
                    scale = np.random.uniform(1, self.opt["resize_range2"][1])
                elif updown_type == "down":
                    scale = np.random.uniform(self.opt["resize_range2"][0], 1)
                else:
                    scale = 1
                mode = random.choice(self.resize_mode_list)
                out = F.interpolate(
                    out,
                    size=(
                        int(ori_h / self.opt["scale"] * scale),
                        int(ori_w / self.opt["scale"] * scale),
                    ),
                    mode=mode,
                )
                normalized_scale = (scale - self.opt["resize_range2"][0]) / (
                    self.opt["resize_range2"][1] - self.opt["resize_range2"][0]
                )
                onehot_mode = torch.zeros(len(self.resize_mode_list))
                for index, mode_current in enumerate(self.resize_mode_list):
                    if mode_current == mode:
                        onehot_mode[index] = 1
                self.degradation_params[
                    :, self.road_map[1] + 4 : self.road_map[1] + 5
                ] = torch.tensor(normalized_scale).expand(self.gt.size(0), 1)
                self.degradation_params[
                    :, self.road_map[1] + 5 : self.road_map[2]
                ] = onehot_mode.expand(self.gt.size(0), len(self.resize_mode_list))
                # noise
                gray_noise_prob = self.opt["gray_noise_prob2"]
                if np.random.uniform() < self.opt["gaussian_noise_prob2"]:
                    (
                        sigma,
                        gray_noise,
                        out,
                        self.noise_g_second,
                    ) = random_add_gaussian_noise_pt_info(
                        out,
                        sigma_range=self.opt["noise_range2"],
                        clip=True,
                        rounds=False,
                        gray_prob=gray_noise_prob,
                    )
                    normalized_sigma = (sigma - self.opt["noise_range2"][0]) / (
                        self.opt["noise_range2"][1] - self.opt["noise_range2"][0]
                    )
                    self.degradation_params[
                        :, self.road_map[2] + 4 : self.road_map[2] + 5
                    ] = normalized_sigma.unsqueeze(1)
                    self.degradation_params[
                        :, self.road_map[2] + 5 : self.road_map[2] + 6
                    ] = gray_noise.unsqueeze(1)
                    self.degradation_params[
                        :, self.road_map[2] + 6 : self.road_map[3]
                    ] = torch.tensor([1, 0]).expand(self.gt.size(0), 2)
                    self.noise_p_second = only_generate_poisson_noise_pt_info(
                        out,
                        scale_range=self.opt["poisson_scale_range2"],
                        gray_prob=gray_noise_prob,
                    )
                else:
                    (
                        scale,
                        gray_noise,
                        out,
                        self.noise_p_second,
                    ) = random_add_poisson_noise_pt_info(
                        out,
                        scale_range=self.opt["poisson_scale_range2"],
                        gray_prob=gray_noise_prob,
                        clip=True,
                        rounds=False,
                    )
                    normalized_scale = (scale - self.opt["poisson_scale_range2"][0]) / (
                        self.opt["poisson_scale_range2"][1]
                        - self.opt["poisson_scale_range2"][0]
                    )
                    self.degradation_params[
                        :, self.road_map[2] + 4 : self.road_map[2] + 5
                    ] = normalized_scale.unsqueeze(1)
                    self.degradation_params[
                        :, self.road_map[2] + 5 : self.road_map[2] + 6
                    ] = gray_noise.unsqueeze(1)
                    self.degradation_params[
                        :, self.road_map[2] + 6 : self.road_map[3]
                    ] = torch.tensor([0, 1]).expand(self.gt.size(0), 2)
                    self.noise_g_second = only_generate_gaussian_noise_pt_info(
                        out,
                        sigma_range=self.opt["noise_range2"],
                        gray_prob=gray_noise_prob,
                    )

                # JPEG compression + the final sinc filter
                if np.random.uniform() < 0.5:
                    # resize back + the final sinc filter
                    mode = random.choice(self.resize_mode_list)
                    onehot_mode = torch.zeros(len(self.resize_mode_list))
                    for index, mode_current in enumerate(self.resize_mode_list):
                        if mode_current == mode:
                            onehot_mode[index] = 1
                    out = F.interpolate(
                        out,
                        size=(ori_h // self.opt["scale"], ori_w // self.opt["scale"]),
                        mode=mode,
                    )
                    out = filter2D(out, self.sinc_kernel)
                    # JPEG compression
                    jpeg_p = out.new_zeros(out.size(0)).uniform_(
                        *self.opt["jpeg_range2"]
                    )
                    normalized_jpeg_p = (jpeg_p - self.opt["jpeg_range2"][0]) / (
                        self.opt["jpeg_range2"][1] - self.opt["jpeg_range2"][0]
                    )
                    out = torch.clamp(out, 0, 1)
                    out = self.jpeger(out, quality=jpeg_p)
                    self.degradation_params[
                        :, self.road_map[3] + 1 : self.road_map[3] + 2
                    ] = normalized_jpeg_p.unsqueeze(1)
                    self.degradation_params[
                        :, self.road_map[3] + 2 : self.road_map[3] + 4
                    ] = torch.tensor([1, 0]).expand(self.gt.size(0), 2)
                    self.degradation_params[
                        :, self.road_map[3] + 4 :
                    ] = onehot_mode.expand(self.gt.size(0), len(self.resize_mode_list))
                else:
                    # JPEG compression
                    jpeg_p = out.new_zeros(out.size(0)).uniform_(
                        *self.opt["jpeg_range2"]
                    )
                    normalized_jpeg_p = (jpeg_p - self.opt["jpeg_range2"][0]) / (
                        self.opt["jpeg_range2"][1] - self.opt["jpeg_range2"][0]
                    )
                    out = torch.clamp(out, 0, 1)
                    out = self.jpeger(out, quality=jpeg_p)
                    # resize back + the final sinc filter
                    mode = random.choice(self.resize_mode_list)
                    onehot_mode = torch.zeros(len(self.resize_mode_list))
                    for index, mode_current in enumerate(self.resize_mode_list):
                        if mode_current == mode:
                            onehot_mode[index] = 1
                    out = F.interpolate(
                        out,
                        size=(ori_h // self.opt["scale"], ori_w // self.opt["scale"]),
                        mode=mode,
                    )
                    out = filter2D(out, self.sinc_kernel)
                    self.degradation_params[
                        :, self.road_map[3] + 1 : self.road_map[3] + 2
                    ] = normalized_jpeg_p.unsqueeze(1)
                    self.degradation_params[
                        :, self.road_map[3] + 2 : self.road_map[3] + 4
                    ] = torch.tensor([0, 1]).expand(self.gt.size(0), 2)
                    self.degradation_params[
                        :, self.road_map[3] + 4 :
                    ] = onehot_mode.expand(self.gt.size(0), len(self.resize_mode_list))

                self.degradation_params = self.degradation_params.to(self.device)

                # clamp and round
                self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.0

                # random crop
                gt_size = self.opt["gt_size"]
                (
                    (self.gt, self.gt_usm),
                    self.lq,
                    self.top,
                    self.left,
                ) = paired_random_crop_return_indexes(
                    [self.gt, self.gt_usm], self.lq, gt_size, self.opt["scale"]
                )

            elif self.degradation_degree == "standard_degrade_one_stage":
                self.degradation_params = torch.zeros(
                    self.opt_train["batch_size_per_gpu"], self.num_degradation_params
                )

                self.kernel1 = data["kernel1"]["kernel"].to(self.device)

                kernel_size_range1 = [
                    self.opt_train["blur_kernel_size_minimum_standard1"],
                    self.opt_train["blur_kernel_size_standard1"],
                ]
                rotation_range = [-math.pi, math.pi]

                self.degradation_params[:, self.road_map[0] : self.road_map[0] + 1] = (
                    data["kernel1"]["kernel_size"].unsqueeze(1) - kernel_size_range1[0]
                ) / (kernel_size_range1[1] - kernel_size_range1[0])
                self.degradation_params[
                    :, self.road_map[0] + 1 : self.road_map[0] + 2
                ] = (
                    data["kernel1"]["sigma_x"].unsqueeze(1)
                    - self.opt_train["blur_sigma_standard1"][0]
                ) / (
                    self.opt_train["blur_sigma_standard1"][1]
                    - self.opt_train["blur_sigma_standard1"][0]
                )
                self.degradation_params[
                    :, self.road_map[0] + 2 : self.road_map[0] + 3
                ] = (
                    data["kernel1"]["sigma_y"].unsqueeze(1)
                    - self.opt_train["blur_sigma_standard1"][0]
                ) / (
                    self.opt_train["blur_sigma_standard1"][1]
                    - self.opt_train["blur_sigma_standard1"][0]
                )
                self.degradation_params[
                    :, self.road_map[0] + 3 : self.road_map[0] + 4
                ] = (data["kernel1"]["rotation"].unsqueeze(1) - rotation_range[0]) / (
                    rotation_range[1] - rotation_range[0]
                )

                ori_h, ori_w = self.gt.size()[2:4]

                # blur
                out = filter2D(self.gt_usm, self.kernel1)
                # random resize
                updown_type = random.choices(
                    ["up", "down", "keep"], self.opt["resize_prob_standard1"]
                )[0]
                if updown_type == "up":
                    scale = np.random.uniform(1, self.opt["resize_range_standard1"][1])
                elif updown_type == "down":
                    scale = np.random.uniform(self.opt["resize_range_standard1"][0], 1)
                else:
                    scale = 1
                mode = random.choice(self.resize_mode_list)
                out = F.interpolate(out, scale_factor=scale, mode=mode)
                normalized_scale = (scale - self.opt["resize_range_standard1"][0]) / (
                    self.opt["resize_range_standard1"][1]
                    - self.opt["resize_range_standard1"][0]
                )
                onehot_mode = torch.zeros(len(self.resize_mode_list))
                for index, mode_current in enumerate(self.resize_mode_list):
                    if mode_current == mode:
                        onehot_mode[index] = 1
                self.degradation_params[
                    :, self.road_map[1] : self.road_map[1] + 1
                ] = torch.tensor(normalized_scale).expand(self.gt.size(0), 1)
                self.degradation_params[
                    :, self.road_map[1] + 1 : self.road_map[1] + 4
                ] = onehot_mode.expand(self.gt.size(0), len(self.resize_mode_list))
                # noise # noise_range: [1, 30] poisson_scale_range: [0.05, 3]
                gray_noise_prob = self.opt["gray_noise_prob_standard1"]
                if np.random.uniform() < self.opt["gaussian_noise_prob_standard1"]:
                    (
                        sigma,
                        gray_noise,
                        out,
                        self.noise_g_first,
                    ) = random_add_gaussian_noise_pt_info(
                        out,
                        sigma_range=self.opt["noise_range_standard1"],
                        clip=True,
                        rounds=False,
                        gray_prob=gray_noise_prob,
                    )

                    normalized_sigma = (
                        sigma - self.opt["noise_range_standard1"][0]
                    ) / (
                        self.opt["noise_range_standard1"][1]
                        - self.opt["noise_range_standard1"][0]
                    )
                    self.degradation_params[
                        :, self.road_map[2] : self.road_map[2] + 1
                    ] = normalized_sigma.unsqueeze(1)
                    self.degradation_params[
                        :, self.road_map[2] + 1 : self.road_map[2] + 2
                    ] = gray_noise.unsqueeze(1)
                    self.degradation_params[
                        :, self.road_map[2] + 2 : self.road_map[2] + 4
                    ] = torch.tensor([1, 0]).expand(self.gt.size(0), 2)
                    self.noise_p_first = only_generate_poisson_noise_pt_info(
                        out,
                        scale_range=self.opt["poisson_scale_range_standard1"],
                        gray_prob=gray_noise_prob,
                    )
                else:
                    (
                        scale,
                        gray_noise,
                        out,
                        self.noise_p_first,
                    ) = random_add_poisson_noise_pt_info(
                        out,
                        scale_range=self.opt["poisson_scale_range_standard1"],
                        gray_prob=gray_noise_prob,
                        clip=True,
                        rounds=False,
                    )
                    normalized_scale = (
                        scale - self.opt["poisson_scale_range_standard1"][0]
                    ) / (
                        self.opt["poisson_scale_range_standard1"][1]
                        - self.opt["poisson_scale_range_standard1"][0]
                    )
                    self.degradation_params[
                        :, self.road_map[2] : self.road_map[2] + 1
                    ] = normalized_scale.unsqueeze(1)
                    self.degradation_params[
                        :, self.road_map[2] + 1 : self.road_map[2] + 2
                    ] = gray_noise.unsqueeze(1)
                    self.degradation_params[
                        :, self.road_map[2] + 2 : self.road_map[2] + 4
                    ] = torch.tensor([0, 1]).expand(self.gt.size(0), 2)
                    self.noise_g_first = only_generate_gaussian_noise_pt_info(
                        out,
                        sigma_range=self.opt["noise_range_standard1"],
                        gray_prob=gray_noise_prob,
                    )

                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(
                    *self.opt["jpeg_range_standard1"]
                )  # tensor([61.6463, 94.2723, 37.1205, 34.9564], device='cuda:0')]
                normalized_jpeg_p = (jpeg_p - self.opt["jpeg_range_standard1"][0]) / (
                    self.opt["jpeg_range_standard1"][1]
                    - self.opt["jpeg_range_standard1"][0]
                )
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
                self.degradation_params[
                    :, self.road_map[3] : self.road_map[3] + 1
                ] = normalized_jpeg_p.unsqueeze(1)

                # resize back
                mode = random.choice(self.resize_mode_list)
                onehot_mode = torch.zeros(len(self.resize_mode_list))
                for index, mode_current in enumerate(self.resize_mode_list):
                    if mode_current == mode:
                        onehot_mode[index] = 1
                out = F.interpolate(
                    out,
                    size=(ori_h // self.opt["scale"], ori_w // self.opt["scale"]),
                    mode=mode,
                )
                self.degradation_params[:, self.road_map[3] + 4 :] = onehot_mode.expand(
                    self.gt.size(0), len(self.resize_mode_list)
                )

                self.degradation_params = self.degradation_params.to(self.device)

                # clamp and round
                self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.0

                # random crop
                gt_size = self.opt["gt_size"]
                (
                    (self.gt, self.gt_usm),
                    self.lq,
                    self.top,
                    self.left,
                ) = paired_random_crop_return_indexes(
                    [self.gt, self.gt_usm], self.lq, gt_size, self.opt["scale"]
                )

            elif self.degradation_degree == "weak_degrade_one_stage":
                self.degradation_params = torch.zeros(
                    self.opt_train["batch_size_per_gpu"], self.num_degradation_params
                )

                self.kernel1 = data["kernel1"]["kernel"].to(self.device)

                kernel_size_range1 = [
                    self.opt_train["blur_kernel_size_minimum_weak1"],
                    self.opt_train["blur_kernel_size_weak1"],
                ]
                rotation_range = [-math.pi, math.pi]

                self.degradation_params[:, self.road_map[0] : self.road_map[0] + 1] = (
                    data["kernel1"]["kernel_size"].unsqueeze(1) - kernel_size_range1[0]
                ) / (kernel_size_range1[1] - kernel_size_range1[0])
                self.degradation_params[
                    :, self.road_map[0] + 1 : self.road_map[0] + 2
                ] = (
                    data["kernel1"]["sigma_x"].unsqueeze(1)
                    - self.opt_train["blur_sigma_weak1"][0]
                ) / (
                    self.opt_train["blur_sigma_weak1"][1]
                    - self.opt_train["blur_sigma_weak1"][0]
                )
                self.degradation_params[
                    :, self.road_map[0] + 2 : self.road_map[0] + 3
                ] = (
                    data["kernel1"]["sigma_y"].unsqueeze(1)
                    - self.opt_train["blur_sigma_weak1"][0]
                ) / (
                    self.opt_train["blur_sigma_weak1"][1]
                    - self.opt_train["blur_sigma_weak1"][0]
                )
                self.degradation_params[
                    :, self.road_map[0] + 3 : self.road_map[0] + 4
                ] = (data["kernel1"]["rotation"].unsqueeze(1) - rotation_range[0]) / (
                    rotation_range[1] - rotation_range[0]
                )

                ori_h, ori_w = self.gt.size()[2:4]

                # blur
                out = filter2D(self.gt_usm, self.kernel1)
                # random resize
                updown_type = random.choices(
                    ["up", "down", "keep"], self.opt["resize_prob_weak1"]
                )[0]
                if updown_type == "up":
                    scale = np.random.uniform(1, self.opt["resize_range_weak1"][1])
                elif updown_type == "down":
                    scale = np.random.uniform(self.opt["resize_range_weak1"][0], 1)
                else:
                    scale = 1
                mode = random.choice(self.resize_mode_list)
                out = F.interpolate(out, scale_factor=scale, mode=mode)
                normalized_scale = (scale - self.opt["resize_range_weak1"][0]) / (
                    self.opt["resize_range_weak1"][1]
                    - self.opt["resize_range_weak1"][0]
                )
                onehot_mode = torch.zeros(len(self.resize_mode_list))
                for index, mode_current in enumerate(self.resize_mode_list):
                    if mode_current == mode:
                        onehot_mode[index] = 1
                self.degradation_params[
                    :, self.road_map[1] : self.road_map[1] + 1
                ] = torch.tensor(normalized_scale).expand(self.gt.size(0), 1)
                self.degradation_params[
                    :, self.road_map[1] + 1 : self.road_map[1] + 4
                ] = onehot_mode.expand(self.gt.size(0), len(self.resize_mode_list))
                # noise # noise_range: [1, 30] poisson_scale_range: [0.05, 3]
                gray_noise_prob = self.opt["gray_noise_prob_weak1"]
                if np.random.uniform() < self.opt["gaussian_noise_prob_weak1"]:
                    (
                        sigma,
                        gray_noise,
                        out,
                        self.noise_g_first,
                    ) = random_add_gaussian_noise_pt_info(
                        out,
                        sigma_range=self.opt["noise_range_weak1"],
                        clip=True,
                        rounds=False,
                        gray_prob=gray_noise_prob,
                    )

                    normalized_sigma = (sigma - self.opt["noise_range_weak1"][0]) / (
                        self.opt["noise_range_weak1"][1]
                        - self.opt["noise_range_weak1"][0]
                    )
                    self.degradation_params[
                        :, self.road_map[2] : self.road_map[2] + 1
                    ] = normalized_sigma.unsqueeze(1)
                    self.degradation_params[
                        :, self.road_map[2] + 1 : self.road_map[2] + 2
                    ] = gray_noise.unsqueeze(1)
                    self.degradation_params[
                        :, self.road_map[2] + 2 : self.road_map[2] + 4
                    ] = torch.tensor([1, 0]).expand(self.gt.size(0), 2)
                    self.noise_p_first = only_generate_poisson_noise_pt_info(
                        out,
                        scale_range=self.opt["poisson_scale_range_weak1"],
                        gray_prob=gray_noise_prob,
                    )
                else:
                    (
                        scale,
                        gray_noise,
                        out,
                        self.noise_p_first,
                    ) = random_add_poisson_noise_pt_info(
                        out,
                        scale_range=self.opt["poisson_scale_range_weak1"],
                        gray_prob=gray_noise_prob,
                        clip=True,
                        rounds=False,
                    )
                    normalized_scale = (
                        scale - self.opt["poisson_scale_range_weak1"][0]
                    ) / (
                        self.opt["poisson_scale_range_weak1"][1]
                        - self.opt["poisson_scale_range_weak1"][0]
                    )
                    self.degradation_params[
                        :, self.road_map[2] : self.road_map[2] + 1
                    ] = normalized_scale.unsqueeze(1)
                    self.degradation_params[
                        :, self.road_map[2] + 1 : self.road_map[2] + 2
                    ] = gray_noise.unsqueeze(1)
                    self.degradation_params[
                        :, self.road_map[2] + 2 : self.road_map[2] + 4
                    ] = torch.tensor([0, 1]).expand(self.gt.size(0), 2)
                    self.noise_g_first = only_generate_gaussian_noise_pt_info(
                        out,
                        sigma_range=self.opt["noise_range_weak1"],
                        gray_prob=gray_noise_prob,
                    )

                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(
                    *self.opt["jpeg_range_weak1"]
                )
                normalized_jpeg_p = (jpeg_p - self.opt["jpeg_range_weak1"][0]) / (
                    self.opt["jpeg_range_weak1"][1] - self.opt["jpeg_range_weak1"][0]
                )
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
                self.degradation_params[
                    :, self.road_map[3] : self.road_map[3] + 1
                ] = normalized_jpeg_p.unsqueeze(1)

                # resize back
                mode = random.choice(self.resize_mode_list)
                onehot_mode = torch.zeros(len(self.resize_mode_list))
                for index, mode_current in enumerate(self.resize_mode_list):
                    if mode_current == mode:
                        onehot_mode[index] = 1
                out = F.interpolate(
                    out,
                    size=(ori_h // self.opt["scale"], ori_w // self.opt["scale"]),
                    mode=mode,
                )
                self.degradation_params[:, self.road_map[3] + 4 :] = onehot_mode.expand(
                    self.gt.size(0), len(self.resize_mode_list)
                )

                self.degradation_params = self.degradation_params.to(self.device)

                # clamp and round
                self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.0

                # random crop
                gt_size = self.opt["gt_size"]
                (
                    (self.gt, self.gt_usm),
                    self.lq,
                    self.top,
                    self.left,
                ) = paired_random_crop_return_indexes(
                    [self.gt, self.gt_usm], self.lq, gt_size, self.opt["scale"]
                )

            else:
                print("Degradation degree mode mismatch.")

            # training pair pool
            self._dequeue_and_enqueue()
            # sharpen self.gt again, as we have changed the self.gt with self._dequeue_and_enqueue
            self.gt_usm = self.usm_sharpener(self.gt)
            self.lq = (
                self.lq.contiguous()
            )  # for the warning: grad and param do not obey the gradient layout contract
        else:
            # for paired training or validation
            data = data_all
            self.lq = data["lq"].to(self.device)
            if "gt" in data:
                self.gt = data["gt"].to(self.device)
                self.gt_usm = self.usm_sharpener(self.gt)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # do not use the synthetic process during validation
        self.is_train = False
        super(MultiDualStarSRGANModel, self).nondist_validation(
            dataloader, current_iter, tb_logger, save_img
        )
        self.is_train = True

    def optimize_parameters(self, current_iter):
        # usm sharpening
        l1_gt = self.gt_usm
        percep_gt = self.gt_usm
        gan_gt = self.gt_usm
        if self.opt["l1_gt_usm"] is False:
            l1_gt = self.gt
        if self.opt["percep_gt_usm"] is False:
            percep_gt = self.gt
        if self.opt["gan_gt_usm"] is False:
            gan_gt = self.gt

        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_g_total = 0
        loss_dict = OrderedDict()
        if (
            current_iter % self.net_d_iters == 0
            and current_iter > self.net_d_init_iters
        ):
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, l1_gt)
                l_g_total += l_g_pix
                loss_dict["l_g_pix"] = l_g_pix
            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style, l_resnet, l_vgg = self.cri_perceptual(
                    self.output, percep_gt
                )
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict["l_g_percep"] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict["l_g_style"] = l_g_style
                if l_resnet is not None:
                    loss_dict["l_g_resnet"] = l_resnet
                if l_vgg is not None:
                    loss_dict["l_g_vgg"] = l_vgg
            # gan loss
            fake_g_preds = self.net_d(self.output)
            loss_dict["l_g_gan"] = 0
            for fake_g_pred in fake_g_preds:
                l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
                l_g_total += l_g_gan
                loss_dict["l_g_gan"] += l_g_gan

            l_g_total.backward()
            self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        # real
        real_d_preds = self.net_d(gan_gt)
        loss_dict["l_d_real"] = 0
        loss_dict["out_d_real"] = 0
        l_d_real_tot = 0
        for real_d_pred in real_d_preds:
            l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
            l_d_real_tot += l_d_real
            loss_dict["l_d_real"] += l_d_real
            loss_dict["out_d_real"] += torch.mean(real_d_pred.detach())
        l_d_real_tot.backward()
        # fake
        loss_dict["l_d_fake"] = 0
        loss_dict["out_d_fake"] = 0
        l_d_fake_tot = 0
        fake_d_preds = self.net_d(self.output.detach().clone())  # clone for pt1.9
        for fake_d_pred in fake_d_preds:
            l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
            l_d_fake_tot += l_d_fake
            loss_dict["l_d_fake"] += l_d_fake
            loss_dict["out_d_fake"] += torch.mean(fake_d_pred.detach())
        l_d_fake_tot.backward()
        self.optimizer_d.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(loss_dict)
