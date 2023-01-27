import os
import random
import zipfile
import findfile

for z_file in findfile.find_cwd_files(and_key=['.zip'],
                                      exclude_key=['.ignore', 'git', 'SuperResolutionAnimeDiffusion'],
                                      recursive=1):
    with zipfile.ZipFile(z_file, 'r') as zip_ref:
        zip_ref.extractall()

import PIL.Image
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
from RealESRGANv030.interface import realEsrgan

magnifier = ImageMagnifier()

start_time = time.time()
is_colab = utils.is_google_colab()

CUDA_VISIBLE_DEVICES = ""
device = autocuda.auto_cuda()

dtype = torch.float16 if device != "cpu" else torch.float32



class Model:
    def __init__(self, name, path="", prefix=""):
        self.name = name
        self.path = path
        self.prefix = prefix
        self.pipe_t2i = None
        self.pipe_i2i = None


models = [
    # Model("anything v3", "Linaqruf/anything-v3.0", "anything v3 style"),
    Model("anything v4.5", "andite/anything-v4.0", "anything v4.5 style"),
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
    solver_order=2,
    # lower_order_final=True,
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
        torch_dtype=dtype,
        scheduler=scheduler,
        safety_checker=lambda images, clip_input: (images, False),
    )

else:  # download all models
    print(f"{datetime.datetime.now()} Downloading vae...")
    vae = AutoencoderKL.from_pretrained(
        current_model.path, subfolder="vae", torch_dtype=dtype
    )
    for model in models:
        try:
            print(f"{datetime.datetime.now()} Downloading {model.name} model...")
            unet = UNet2DConditionModel.from_pretrained(
                model.path, subfolder="unet", torch_dtype=dtype
            )
            model.pipe_t2i = StableDiffusionPipeline.from_pretrained(
                model.path,
                unet=unet,
                vae=vae,
                torch_dtype=dtype,
                scheduler=scheduler,
                safety_checker=None,
            )
            model.pipe_i2i = StableDiffusionImg2ImgPipeline.from_pretrained(
                model.path,
                unet=unet,
                vae=vae,
                torch_dtype=dtype,
                scheduler=scheduler,
                safety_checker=None,
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

# model.pipe_i2i = torch.compile(model.pipe_i2i)
# model.pipe_t2i = torch.compile(model.pipe_t2i)
if torch.cuda.is_available():
    pipe = pipe.to(device)


# device = "GPU 🔥" if torch.cuda.is_available() else "CPU 🥶"


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

    return (
        gr.update(visible=model_name == models[0].name),
        gr.update(placeholder=prefix),
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
    scale="ESRGAN4x",
    scale_factor=2,
):
    fprint(psutil.virtual_memory())  # print memory usage

    fprint(f"Prompt: {prompt}")
    global current_model
    for model in models:
        if model.name == model_name:
            current_model = model
            model_path = current_model.path

    generator = torch.Generator(device).manual_seed(seed) if seed != 0 else None

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
                    scale,
                    scale_factor,
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
                    scale,
                    scale_factor,
                ),
                None,
            )
    except Exception as e:
        return None, error_str(e)
    # if img is not None:
    #     return img_to_img(model_path, prompt, neg_prompt, img, strength, guidance, steps, width, height,
    #                       generator, scale, scale_factor), None
    # else:
    #     return txt_to_img(model_path, prompt, neg_prompt, guidance, steps, width, height, generator, scale, scale_factor), None


def txt_to_img(
    model_path,
    prompt,
    neg_prompt,
    guidance,
    steps,
    width,
    height,
    generator,
    scale,
    scale_factor,
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
                torch_dtype=dtype,
                scheduler=scheduler,
                safety_checker=lambda images, clip_input: (images, False),
            )
        else:
            # pipe = pipe.to("cpu")
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

    # result.images[0] = magnifier.magnify(result.images[0], scale_factor=scale_factor)
    # enhance resolution
    if scale_factor > 1:
        if scale == "ESRGAN4x":
            fp32 = True if device == "cpu" else False
            result.images[0] = realEsrgan(
                input_dir=result.images[0],
                suffix="",
                output_dir="imgs",
                fp32=fp32,
                outscale=scale_factor,
            )[0]
        else:
            result.images[0] = magnifier.magnify(
                result.images[0], scale_factor=scale_factor
            )
    # save image
    result.images[0].save(
        "imgs/result-{}.png".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
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
    scale,
    scale_factor,
):
    fprint(f"{datetime.datetime.now()} img_to_img, model: {model_path}")

    global last_mode
    global pipe
    global current_model_path
    if model_path != current_model_path or last_mode != "img2img":
        current_model_path = model_path

        if is_colab or current_model == custom_model:
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                current_model_path,
                torch_dtype=dtype,
                scheduler=scheduler,
                safety_checker=lambda images, clip_input: (images, False),
            )
        else:
            # pipe = pipe.to("cpu")
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
        image=img,
        num_inference_steps=int(steps),
        strength=strength,
        guidance_scale=guidance,
        # width=width,
        # height=height,
        generator=generator,
    )
    if scale_factor > 1:
        if scale == "ESRGAN4x":
            fp32 = True if device == "cpu" else False
            result.images[0] = realEsrgan(
                input_dir=result.images[0],
                suffix="",
                output_dir="imgs",
                fp32=fp32,
                outscale=scale_factor,
            )[0]
        else:
            result.images[0] = magnifier.magnify(
                result.images[0], scale_factor=scale_factor
            )
    # save image
    result.images[0].save(
        "imgs/result-{}.png".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    )
    return replace_nsfw_images(result)


def replace_nsfw_images(results):
    if is_colab:
        return results.images[0]
    if hasattr(results, "nsfw_content_detected") and results.nsfw_content_detected:
        for i in range(len(results.images)):
            if results.nsfw_content_detected[i]:
                results.images[i] = Image.open("nsfw.png")
    return results.images[0]


css = """.finetuned-diffusion-div div{display:inline-flex;align-items:center;gap:.8rem;font-size:1.75rem}.finetuned-diffusion-div div h1{font-weight:900;margin-bottom:7px}.finetuned-diffusion-div p{margin-bottom:10px;font-size:94%}a{text-decoration:underline}.tabs{margin-top:0;margin-bottom:0}#gallery{min-height:20rem}
"""
with gr.Blocks(css=css) as demo:
    if not os.path.exists("imgs"):
        os.mkdir("imgs")

    gr.Markdown("# Super Resolution Anime Diffusion")
    gr.Markdown(
        "## Author: [yangheng95](https://github.com/yangheng95)  Github:[Github](https://github.com/yangheng95/SuperResolutionAnimeDiffusion)"
    )
    gr.Markdown(
        "### This demo is running on a CPU, so it will take at least 20 minutes. "
        "If you have a GPU, you can clone from [Github](https://github.com/yangheng95/SuperResolutionAnimeDiffusion) and run it locally."
    )
    gr.Markdown(
        "### FYI: to generate a 512*512 image and magnify 4x, it only takes 5~8 seconds on a RTX 2080 GPU"
    )
    gr.Markdown(
        "### You can duplicate this demo on HuggingFace Spaces, click [here](https://huggingface.co/spaces/yangheng/Super-Resolution-Anime-Diffusion?duplicate=true)"
    )

    with gr.Row():
        with gr.Column(scale=55):
            with gr.Group():
                gr.Markdown("Text to image")

                model_name = gr.Dropdown(
                    label="Model",
                    choices=[m.name for m in models],
                    value=current_model.name,
                )

                with gr.Box(visible=False) as custom_model_group:
                    custom_model_path = gr.Textbox(
                        label="Custom model path",
                        placeholder="Path to model, e.g. nitrosocke/Arcane-Diffusion",
                        interactive=True,
                    )
                    gr.HTML(
                        "<div><font size='2'>Custom models have to be downloaded first, so give it some time.</font></div>"
                    )

                with gr.Row():
                    prompt = gr.Textbox(
                        label="Prompt",
                        show_label=False,
                        max_lines=2,
                        placeholder="Enter prompt. Style applied automatically",
                    ).style(container=False)
                with gr.Row():
                    generate = gr.Button(value="Generate")

                with gr.Row():
                    with gr.Group():
                        neg_prompt = gr.Textbox(
                            label="Negative prompt",
                            value="bad result, worst, random, invalid, inaccurate, imperfect, blurry, deformed,"
                                  " disfigured, mutation, mutated, ugly, out of focus, bad anatomy, text, error,"
                                  " extra digit, fewer digits, worst quality, low quality, normal quality, noise, "
                                  "jpeg artifact, compression artifact, signature, watermark, username, logo, "
                                  "low resolution, worst resolution, bad resolution, normal resolution, bad detail,"
                                  " bad details, bad lighting, bad shadow, bad shading, bad background,"
                                  " worst background.",
                        )

                image_out = gr.Image(height="auto", width="auto")
                error_output = gr.Markdown()

                with gr.Row():
                    gr.Markdown(
                        "# Random Image Generation Preview (512*768)x4 magnified"
                    )
                for f_img in findfile.find_cwd_files(".png", recursive=2):
                    with gr.Row():
                        image = gr.Image(height=512, value=PIL.Image.open(f_img))
                # gallery = gr.Gallery(
                #     label="Generated images", show_label=False, elem_id="gallery"
                # ).style(grid=[1], height="auto")

        with gr.Column(scale=45):
            with gr.Group():
                gr.Markdown("Image to Image")

                with gr.Row():
                    with gr.Group():
                        image = gr.Image(
                            label="Image", height=256, tool="editor", type="pil"
                        )
                        strength = gr.Slider(
                            label="Transformation strength",
                            minimum=0,
                            maximum=1,
                            step=0.01,
                            value=0.5,
                        )

                with gr.Row():
                    with gr.Group():
                        # n_images = gr.Slider(label="Images", value=1, minimum=1, maximum=4, step=1)

                        with gr.Row():
                            guidance = gr.Slider(
                                label="Guidance scale", value=7.5, maximum=15
                            )
                            steps = gr.Slider(
                                label="Steps", value=15, minimum=2, maximum=75, step=1
                            )

                        with gr.Row():
                            width = gr.Slider(
                                label="Width",
                                value=512,
                                minimum=64,
                                maximum=1024,
                                step=8,
                            )
                            height = gr.Slider(
                                label="Height",
                                value=768,
                                minimum=64,
                                maximum=1024,
                                step=8,
                            )
                        with gr.Row():
                            scale = gr.Radio(
                                label="Scale",
                                choices=["Waifu2x", "ESRGAN4x"],
                                value="Waifu2x",
                            )
                        with gr.Row():
                            scale_factor = gr.Slider(
                                1,
                                8,
                                label="Scale factor (to magnify image) (1, 2, 4, 8)",
                                value=1,
                                step=1,
                            )

                        seed = gr.Slider(
                            0, 2147483647, label="Seed (0 = random)", value=0, step=1
                        )

    if is_colab:
        model_name.change(
            on_model_change,
            inputs=model_name,
            outputs=[custom_model_group, prompt],
            queue=False,
        )
        custom_model_path.change(
            custom_model_changed, inputs=custom_model_path, outputs=None
        )
    # n_images.change(lambda n: gr.Gallery().style(grid=[2 if n > 1 else 1], height="auto"), inputs=n_images, outputs=gallery)

    gr.Markdown(
        "### based on [Anything V4.5] and [Anything V3](https://huggingface.co/Linaqruf/anything-v3.0)"
    )

    inputs = [
        model_name,
        prompt,
        guidance,
        steps,
        width,
        height,
        seed,
        image,
        strength,
        neg_prompt,
        scale,
        scale_factor,
    ]
    outputs = [image_out, error_output]
    prompt.submit(inference, inputs=inputs, outputs=outputs)
    generate.click(inference, inputs=inputs, outputs=outputs, api_name="generate")

    prompt_keys = [
        "girl",
        "lovely",
        "cute",
        "beautiful eyes",
        "cumulonimbus clouds",
        random.choice(["dress"]),
        random.choice(["white hair"]),
        random.choice(["blue eyes"]),
        random.choice(["flower meadow"]),
        random.choice(["Elif", "Angel"]),
    ]
    prompt.value = ",".join(prompt_keys)
    ex = gr.Examples(
        [
            [models[0].name, prompt.value, 7.5, 15],
        ],
        inputs=[model_name, prompt, guidance, steps, seed],
        outputs=outputs,
        fn=inference,
        cache_examples=False,
    )

print(f"Space built in {time.time() - start_time:.2f} seconds")

if not is_colab:
    demo.queue(concurrency_count=2)
demo.launch(debug=is_colab, enable_queue=True, share=is_colab)
