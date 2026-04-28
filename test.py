import os
import cv2
import time
import random
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms
from accelerate.utils import set_seed


def image_process(args, content_image=None, style_image=None):
    if not args.demo:
        # Read content image and style image
        if args.character_input:
            assert args.content_character is not None, "The content_character should not be None."
            if not is_char_in_font(font_path=args.ttf_path, char=args.content_character):
                return None, None
            font = load_ttf(ttf_path=args.ttf_path)
            content_image = ttf2im(font=font, char=args.content_character)
            content_image_pil = content_image.copy()
        else:
            content_image = Image.open(args.content_image_path).convert('RGB')
            content_image_pil = None
        style_image = Image.open(args.style_image_path).convert('RGB')
    else:
        assert style_image is not None, "The style image should not be None."
        if args.character_input:
            assert args.content_character is not None, "The content_character should not be None."
            if not is_char_in_font(font_path=args.ttf_path, char=args.content_character):
                return None, None
            font = load_ttf(ttf_path=args.ttf_path)
            content_image = ttf2im(font=font, char=args.content_character)
        else:
            assert content_image is not None, "The content image should not be None."
        content_image_pil = None
        
    ## Dataset transform
    content_inference_transforms = transforms.Compose(
        [transforms.Resize(args.content_image_size, \
                            interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])])
    style_inference_transforms = transforms.Compose(
        [transforms.Resize(args.style_image_size, \
                           interpolation=transforms.InterpolationMode.BILINEAR),
         transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    content_image = content_inference_transforms(content_image)[None, :]
    style_image = style_inference_transforms(style_image)[None, :]

    return content_image, style_image, content_image_pil


def testing(args, pipe, content_image=None, style_image=None):
    if not args.demo:
        os.makedirs(args.save_image_dir, exist_ok=True)
        # saving sampling config
        save_args_to_yaml(args=args, output_file=f"{args.save_image_dir}/sampling_config.yaml")

    if args.seed:
        set_seed(seed=args.seed)
    
    content_image, style_image, content_image_pil = image_process(args=args, 
                                                                  content_image=content_image, 
                                                                  style_image=style_image)
    if content_image == None:
        print(f"The content_character you provided is not in the ttf. \
                Please change the content_character or you can change the ttf.")
        return None

    with torch.no_grad():
        content_image = content_image.to(args.device)
        style_image = style_image.to(args.device)
        start = time.time()
        images = pipe.generate(
            content_images=content_image,
            style_images=style_image,
            batch_size=1,
            order=args.order,
            num_inference_step=args.num_inference_steps,
            content_encoder_downsample_size=args.content_encoder_downsample_size,
            skip_type=args.skip_type,
            method=args.method,
            correcting_x0_fn=args.correcting_x0_fn)
        end = time.time()

        if args.save_image:
            print(f"Saving the image ......")
            save_single_image(save_dir=args.save_image_dir, image=images[0])
            if args.character_input:
                save_image_with_content_style(save_dir=args.save_image_dir,
                                            image=images[0],
                                            content_image_pil=content_image_pil,
                                            content_image_path=None,
                                            style_image_path=args.style_image_path,
                                            resolution=args.resolution)
            else:
                save_image_with_content_style(save_dir=args.save_image_dir,
                                            image=images[0],
                                            content_image_pil=None,
                                            content_image_path=args.content_image_path,
                                            style_image_path=args.style_image_path,
                                            resolution=args.resolution)
            print(f"Finish the sampling process, costing time {end - start}s")
        return images[0]


def load_controlnet_pipeline(args,
                             config_path="lllyasviel/sd-controlnet-canny", 
                             ckpt_path="runwayml/stable-diffusion-v1-5"):
    from diffusers import ControlNetModel, AutoencoderKL
    # load controlnet model and pipeline
    from diffusers import StableDiffusionControlNetPipeline, UniPCMultistepScheduler
    controlnet = ControlNetModel.from_pretrained(config_path, 
                                                 torch_dtype=torch.float16,
                                                 cache_dir=f"{args.ckpt_dir}/controlnet")
    print(f"Loaded ControlNet Model Successfully!")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(ckpt_path, 
                                                             controlnet=controlnet, 
                                                             torch_dtype=torch.float16,
                                                             cache_dir=f"{args.ckpt_dir}/controlnet_pipeline")
    # faster
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    print(f"Loaded ControlNet Pipeline Successfully!")

    return pipe




if __name__=="__main__":
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    args = arg_parse()
    
    # load fontdiffuser pipeline
    pipe = load_fontdiffuer_pipeline(args=args)
    out_image = sampling(args=args, pipe=pipe)