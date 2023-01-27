import os
import random

import autocuda
from pyabsa.utils.pyabsa_utils import fprint

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    DPMSolverMultistepScheduler,
)
import gradio as gr
import torch
from PIL import Image
import utils
import datetime
import time
import psutil

from Waifu2x.magnify import ImageMagnifier

start_time = time.time()
is_colab = utils.is_google_colab()

device = autocuda.auto_cuda()

magnifier = ImageMagnifier()


class Model:
    def __init__(self, name, path="", prefix=""):
        self.name = name
        self.path = path
        self.prefix = prefix
        self.pipe_t2i = None
        self.pipe_i2i = None


models = [
    # Model("anything v3", "anything-v3.0", "anything v3 style"),
    Model("anything v3", "Linaqruf/anything-v3.0", "anything v3 style"),
]
#  Model("Spider-Verse", "nitrosocke/spider-verse-diffusion", "spiderverse style "),
#  Model("Balloon Art", "Fictiverse/Stable_Diffusion_BalloonArt_Model", "BalloonArt "),
#  Model("Elden Ring", "nitrosocke/elden-ring-diffusion", "elden ring style "),
#  Model("Tron Legacy", "dallinmackay/Tron-Legacy-diffusion", "trnlgcy ")
# Model("Pokémon", "lambdalabs/sd-pokemon-diffusers", ""),
# Model("Pony Diffusion", "AstraliteHeart/pony-diffusion", ""),
# Model("Robo Diffusion", "nousr/robo-diffusion", ""),

scheduler = DPMSolverMultistepScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    num_train_timesteps=1000,
    trained_betas=None,
    predict_epsilon=True,
    thresholding=False,
    algorithm_type="dpmsolver++",
    solver_type="midpoint",
    lower_order_final=True,
)

custom_model = None
if is_colab:
    models.insert(0, Model("Custom model"))
    custom_model = models[0]

last_mode = "txt2img"
current_model = models[1] if is_colab else models[0]
current_model_path = current_model.path

if is_colab:
    pipe = StableDiffusionPipeline.from_pretrained(
        current_model.path,
        torch_dtype=torch.float16,
        scheduler=scheduler,
        safety_checker=lambda images, clip_input: (images, False),
    )

else:  # download all models
    print(f"{datetime.datetime.now()} Downloading vae...")
    vae = AutoencoderKL.from_pretrained(
        current_model.path, subfolder="vae", torch_dtype=torch.float16
    )
    for model in models:
        try:
            print(f"{datetime.datetime.now()} Downloading {model.name} model...")
            unet = UNet2DConditionModel.from_pretrained(
                model.path, subfolder="unet", torch_dtype=torch.float16
            )
            model.pipe_t2i = StableDiffusionPipeline.from_pretrained(
                model.path,
                unet=unet,
                vae=vae,
                torch_dtype=torch.float16,
                scheduler=scheduler,
            )
            model.pipe_i2i = StableDiffusionImg2ImgPipeline.from_pretrained(
                model.path,
                unet=unet,
                vae=vae,
                torch_dtype=torch.float16,
                scheduler=scheduler,
            )
        except Exception as e:
            print(
                f"{datetime.datetime.now()} Failed to load model "
                + model.name
                + ": "
                + str(e)
            )
            models.remove(model)
    pipe = models[0].pipe_t2i

if torch.cuda.is_available():
    pipe = pipe.to(device)

device = "GPU 🔥" if torch.cuda.is_available() else "CPU 🥶"


def error_str(error, title="Error"):
    return (
        f"""#### {title}
            {error}"""
        if error
        else ""
    )


def custom_model_changed(path):
    models[0].path = path
    global current_model
    current_model = models[0]


def on_model_change(model_name):
    prefix = (
        'Enter prompt. "'
        + next((m.prefix for m in models if m.name == model_name), None)
        + '" is prefixed automatically'
        if model_name != models[0].name
        else "Don't forget to use the custom model prefix in the prompt!"
    )

    return gr.update(visible=model_name == models[0].name), gr.update(
        placeholder=prefix
    )


def inference(
    model_name,
    prompt,
    guidance,
    steps,
    width=512,
    height=512,
    seed=0,
    img=None,
    strength=0.5,
    neg_prompt="",
):
    print(psutil.virtual_memory())  # print memory usage

    global current_model
    for model in models:
        if model.name == model_name:
            current_model = model
            model_path = current_model.path

    generator = torch.Generator("cuda").manual_seed(seed) if seed != 0 else None

    try:
        if img is not None:
            return (
                img_to_img(
                    model_path,
                    prompt,
                    neg_prompt,
                    img,
                    strength,
                    guidance,
                    steps,
                    width,
                    height,
                    generator,
                ),
                None,
            )
        else:
            return (
                txt_to_img(
                    model_path,
                    prompt,
                    neg_prompt,
                    guidance,
                    steps,
                    width,
                    height,
                    generator,
                ),
                None,
            )
    except Exception as e:
        fprint(e)
        return None, error_str(e)


def txt_to_img(
    model_path, prompt, neg_prompt, guidance, steps, width, height, generator
):
    print(f"{datetime.datetime.now()} txt_to_img, model: {current_model.name}")

    global last_mode
    global pipe
    global current_model_path
    if model_path != current_model_path or last_mode != "txt2img":
        current_model_path = model_path

        if is_colab or current_model == custom_model:
            pipe = StableDiffusionPipeline.from_pretrained(
                current_model_path,
                torch_dtype=torch.float16,
                scheduler=scheduler,
                safety_checker=lambda images, clip_input: (images, False),
            )
        else:
            pipe = pipe.to("cpu")
            pipe = current_model.pipe_t2i

        if torch.cuda.is_available():
            pipe = pipe.to(device)
        last_mode = "txt2img"

    prompt = current_model.prefix + prompt
    result = pipe(
        prompt,
        negative_prompt=neg_prompt,
        # num_images_per_prompt=n_images,
        num_inference_steps=int(steps),
        guidance_scale=guidance,
        width=width,
        height=height,
        generator=generator,
    )
    result.images[0] = magnifier.magnify(result.images[0])
    result.images[0] = magnifier.magnify(result.images[0])

    # save image
    result.images[0].save(
        "{}/{}.{}.{}.{}.{}.{}.{}.{}.png".format(
            saved_path,
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
            model_name,
            prompt,
            guidance,
            steps,
            width,
            height,
            seed,
        )
    )
    return replace_nsfw_images(result)


def img_to_img(
    model_path,
    prompt,
    neg_prompt,
    img,
    strength,
    guidance,
    steps,
    width,
    height,
    generator,
):
    print(f"{datetime.datetime.now()} img_to_img, model: {model_path}")

    global last_mode
    global pipe
    global current_model_path
    if model_path != current_model_path or last_mode != "img2img":
        current_model_path = model_path

        if is_colab or current_model == custom_model:
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                current_model_path,
                torch_dtype=torch.float16,
                scheduler=scheduler,
                safety_checker=lambda images, clip_input: (images, False),
            )
        else:
            pipe = pipe.to("cpu")
            pipe = current_model.pipe_i2i

        if torch.cuda.is_available():
            pipe = pipe.to(device)
        last_mode = "img2img"

    prompt = current_model.prefix + prompt
    ratio = min(height / img.height, width / img.width)
    img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
    result = pipe(
        prompt,
        negative_prompt=neg_prompt,
        # num_images_per_prompt=n_images,
        init_image=img,
        num_inference_steps=int(steps),
        strength=strength,
        guidance_scale=guidance,
        width=width,
        height=height,
        generator=generator,
    )
    result.images[0] = magnifier.magnify(result.images[0])
    result.images[0] = magnifier.magnify(result.images[0])

    # save image
    result.images[0].save(
        "{}/{}.{}.{}.{}.{}.{}.{}.{}.png".format(
            saved_path,
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
            model_name,
            prompt,
            guidance,
            steps,
            width,
            height,
            seed,
        )
    )
    return replace_nsfw_images(result)


def replace_nsfw_images(results):
    if is_colab:
        return results.images[0]

    for i in range(len(results.images)):
        if results.nsfw_content_detected[i]:
            results.images[i] = Image.open("nsfw.png")
    return results.images[0]


if __name__ == "__main__":
    # inference("DALL-E", "a dog", 0, 1000, 512, 512, 0, None, 0.5, "")
    model_name = "anything v3"
    saved_path = r"imgs"
    if not os.path.exists(saved_path):
        os.mkdir(saved_path)
    n = 0
    while True:
        prompt_keys = [
            "beautiful eyes",
            "cumulonimbus clouds",
            "sky",
            "detailed fingers",
            random.choice(
                [
                    "white hair",
                    "red hair",
                    "blonde hair",
                    "black hair",
                    "green hair",
                ]
            ),
            random.choice(
                [
                    "blue eyes",
                    "green eyes",
                    "red eyes",
                    "black eyes",
                    "yellow eyes",
                ]
            ),
            random.choice(["flower meadow", "garden", "city", "river", "beach"]),
            random.choice(["Elif", "Angel"]),
        ]
        guidance = 7.5
        steps = 25
        # width = 1024
        # height = 1024
        # width = 768
        # height = 1024
        width = 512
        height = 888
        seed = 0
        img = None
        strength = 0.5
        neg_prompt = ""
        inference(
            model_name,
            ".".join(prompt_keys),
            guidance,
            steps,
            width=width,
            height=height,
            seed=seed,
            img=img,
            strength=strength,
            neg_prompt=neg_prompt,
        )
        n += 1
        fprint(n)
