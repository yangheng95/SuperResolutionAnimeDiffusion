# -*- coding: utf-8 -*-
# file: image_scale.py
# time: 06/12/2022
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2021. All Rights Reserved.
import os

import findfile
import tqdm

from Waifu2x import ImageMagnifier

magnifier = ImageMagnifier()
if __name__ == "__main__":
    # path = os.getcwd()
    # for f in findfile.find_cwd_files(or_key=[".jpg", ".png"]):
    for f in tqdm.tqdm(
        findfile.find_files(r"C:\Users\chuan\OneDrive\imgs", or_key=[".jpg", ".png"])
    ):
        img = magnifier.magnify_from_file(f, scale_factor=2)
