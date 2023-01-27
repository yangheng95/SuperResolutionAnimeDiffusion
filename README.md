---
title: Anything V3.0
emoji: 🏃
colorFrom: gray
colorTo: yellow
sdk: gradio
sdk_version: 3.10.1
app_file: app.py
pinned: false
---

# Super Resolution Anime Diffusion 


# [Online Web Demo](https://huggingface.co/spaces/yangheng/Super-Resolution-Anime-Diffusion)

This is demo forked from https://huggingface.co/Linaqruf/anything-v3.0.

## Super Resolution Anime Diffusion
At this moment, many diffusion models can only generate <1024 width and length pictures.
I integrated the Super Resolution with [Anything diffusion model](https://huggingface.co/Linaqruf/anything-v3.0) to produce high resolution pictures.
Thanks to the open-source project: https://github.com/yu45020/Waifu2x


## Modifications
1. Disable the safety checker to save time and memory. You need to abide the original rules of the model.
2. Add the Super Resolution function to the model.
3. Add batch generation function to the model (see inference.py).

## Install 
1. Install [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. create a conda environment:
```bash
conda create -n diffusion python=3.9
conda activate diffusion
```
3. install requirements:
```ash
conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```
4. Run web demo:
```
python app.py
```
5. or run batch anime-generation
```
python inference.py
```
see the source code for details, you can set scale factor to magnify pictures

## Random Examples (512*768) x4 scale factor
![Anime Girl](./random_examples/1.png)
![Anime Girl](./random_examples/2.png)
# Origin README
---
language:
- en
license: creativeml-openrail-m
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
inference: true
---

# Anything V3

Welcome to Anything V3 - a latent diffusion model for weebs. This model is intended to produce high-quality, highly detailed anime style with just a few prompts. Like other anime-style Stable Diffusion models, it also supports danbooru tags to generate images.

e.g. **_1girl, white hair, golden eyes, beautiful eyes, detail, flower meadow, cumulonimbus clouds, lighting, detailed sky, garden_** 

## Gradio

We support a [Gradio](https://github.com/gradio-app/gradio) Web UI to run Anything-V3.0:

[Open in Spaces](https://huggingface.co/spaces/akhaliq/anything-v3.0)



## 🧨 Diffusers

This model can be used just like any other Stable Diffusion model. For more information,
please have a look at the [Stable Diffusion](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion).

You can also export the model to [ONNX](https://huggingface.co/docs/diffusers/optimization/onnx), [MPS](https://huggingface.co/docs/diffusers/optimization/mps) and/or [FLAX/JAX]().

```python
from diffusers import StableDiffusionPipeline
import torch

model_id = "Linaqruf/anything-v3.0"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "pikachu"
image = pipe(prompt).images[0]

image.save("./pikachu.png")
```

## Examples

Below are some examples of images generated using this model:

**Anime Girl:**
![Anime Girl](https://huggingface.co/Linaqruf/anything-v3.0/resolve/main/1girl.png)
```
1girl, brown hair, green eyes, colorful, autumn, cumulonimbus clouds, lighting, blue sky, falling leaves, garden
Steps: 50, Sampler: DDIM, CFG scale: 12
```
**Anime Boy:**
![Anime Boy](https://huggingface.co/Linaqruf/anything-v3.0/resolve/main/1boy.png)
```
1boy, medium hair, blonde hair, blue eyes, bishounen, colorful, autumn, cumulonimbus clouds, lighting, blue sky, falling leaves, garden
Steps: 50, Sampler: DDIM, CFG scale: 12
```
**Scenery:**
![Scenery](https://huggingface.co/Linaqruf/anything-v3.0/resolve/main/scenery.png)
```
scenery, shibuya tokyo, post-apocalypse, ruins, rust, sky, skyscraper, abandoned, blue sky, broken window, building, cloud, crane machine, outdoors, overgrown, pillar, sunset
Steps: 50, Sampler: DDIM, CFG scale: 12
```

## License

This model is open access and available to all, with a CreativeML OpenRAIL-M license further specifying rights and usage.
The CreativeML OpenRAIL License specifies: 

1. You can't use the model to deliberately produce nor share illegal or harmful outputs or content 
2. The authors claims no rights on the outputs you generate, you are free to use them and are accountable for their use which must not go against the provisions set in the license
3. You may re-distribute the weights and use the model commercially and/or as a service. If you do, please be aware you have to include the same use restrictions as the ones in the license and share a copy of the CreativeML OpenRAIL-M to all your users (please read the license entirely and carefully)
[Please read the full license here](https://huggingface.co/spaces/CompVis/stable-diffusion-license)
