# -*- coding: utf-8 -*-
# file: test.py
# time: 05/12/2022
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2021. All Rights Reserved.
import autocuda
import findfile
from torch.cuda.amp import autocast
from torchvision import transforms
from .utils.prepare_images import *
from .Models import *


class ResolutionMagnifier:

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
    def magnify(self, img):
        # origin
        img_splitter = ImageSplitter(seg_size=64, scale_factor=2, boarder_pad_size=3)
        img_patches = img_splitter.split_img_tensor(img, scale_method=None, img_pad=0)
        with torch.no_grad():
            if self.device!='cpu':
                with autocast():
                    out = [self.model_cran_v2(i.to(self.device)) for i in img_patches]
            else:
                out = [self.model_cran_v2(i).to(self.device) for i in img_patches]
        img_upscale = img_splitter.merge_img_tensor(out)

        final = torch.cat([img_upscale])

        return transforms.ToPILImage()(final[0])
