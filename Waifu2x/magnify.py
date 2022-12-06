# -*- coding: utf-8 -*-
# file: test.py
# time: 05/12/2022
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2021. All Rights Reserved.
from pathlib import Path
from typing import Union

import autocuda
import findfile
from pyabsa.utils.pyabsa_utils import fprint
from torchvision import transforms
from .utils.prepare_images import *
from .Models import *


class ImageMagnifier:

    def __init__(self):
        self.device = autocuda.auto_cuda()
        self.model_cran_v2 = CARN_V2(color_channels=3, mid_channels=64, conv=nn.Conv2d,
                                     single_conv_size=3, single_conv_group=1,
                                     scale=2, activation=nn.LeakyReLU(0.1),
                                     SEBlock=True, repeat_blocks=3, atrous=(1, 1, 1))

        self.model_cran_v2 = network_to_half(self.model_cran_v2)
        self.checkpoint = findfile.find_cwd_file("CARN_model_checkpoint.pt")
        self.model_cran_v2.load_state_dict(torch.load(self.checkpoint, map_location='cpu'))
        # if use GPU, then comment out the next line so it can use fp16.
        self.model_cran_v2 = self.model_cran_v2.float().to(self.device)
        self.model_cran_v2.to(self.device)

    def __image_scale(self, img, scale_factor: int = 2):
        img_splitter = ImageSplitter(seg_size=64, scale_factor=scale_factor, boarder_pad_size=3)
        img_patches = img_splitter.split_img_tensor(img, scale_method=None, img_pad=0)
        with torch.no_grad():
            if self.device != 'cpu':
                with torch.cuda.amp.autocast():
                    out = [self.model_cran_v2(i.to(self.device)) for i in img_patches]
            else:
                with torch.cpu.amp.autocast():
                    out = [self.model_cran_v2(i) for i in img_patches]
        img_upscale = img_splitter.merge_img_tensor(out)

        final = torch.cat([img_upscale])

        return transforms.ToPILImage()(final[0])

    def magnify(self, img, scale_factor: int = 2):
        fprint("scale factor reset to:", scale_factor//2*2)
        _scale_factor = scale_factor
        while _scale_factor // 2 > 0:
            img = self.__image_scale(img, scale_factor=2)
            _scale_factor = _scale_factor // 2
        return img

    def magnify_from_file(self, img_path: Union[str, Path], scale_factor: int = 2, save_img: bool = True):

        if not os.path.exists(img_path):
            raise FileNotFoundError("Path is not found.")
        if os.path.isfile(img_path):
            try:
                img = Image.open(img_path)
                img = self.magnify(img, scale_factor)
                if save_img:
                    img.save(os.path.join(img_path))
            except Exception as e:
                fprint(img_path, e)
            fprint(img_path, "Done.")

        elif os.path.isdir(img_path):
            for path in os.listdir(img_path):
                try:
                    img = Image.open(os.path.join(img_path, path))
                    img = self.magnify(img, scale_factor)
                    if save_img:
                        img.save(os.path.join(img_path, path))
                except Exception as e:
                    fprint(path, e)
                    continue
                fprint(path, "Done.")
        else:
            raise TypeError("Path is not a file or directory.")
