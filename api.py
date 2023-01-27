# -*- coding: utf-8 -*-
# file: api.py.py
# time: 20:37 2022/12/6
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2021. All Rights Reserved.
import requests
from PIL import Image
from io import BytesIO

response = requests.post(
    "https://yangheng-super-resolution-anime-diffusion.hf.space/run/generate",
    json={
        "data": [
            "anything v3",
            "girl,lovely,cute,beautiful eyes,cumulonimbus clouds,sky,detailed fingers,pants,red hair,blue eyes,flower meadow,Elif",
            7.5,
            15,
            512,
            512,
            0,
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg==",
            0.5,
            "",
            2,
        ]
    },
    timeout=3000,
)

img = Image.open(BytesIO(response.content))
img.show()
img.save("test_api.png")
