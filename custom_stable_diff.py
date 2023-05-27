from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler #, KDPM2AncestralDiscreteScheduler, DPMSolverSinglestepScheduler,  KDPM2DiscreteScheduler
import lpw_stable_diffusion
import torch 
import time
import numpy as np
import click
from PIL import Image
import os
import logging
import random_prompt as rp
import sys
STEPS = 28
CFG = 8
WIDTH = 256
HEIGHT = 256
BATCH = 1
SEED = None
PROMPT = rp.generate_prompt()
NEG = "(worst quality, low quality:1.4), (dusty sunbeams:1.0), (motion lines, motion blur:1.4), (greyscale, monochrome:1.0), text, title, logo, signature,nsfw"

model_path = os.path.join(os.path.dirname(__file__),"my_local_model_folder")

#Load the custom pipeline, this bad guy lets us use the long prompt weighting like in the Web UI
LongPromptWeightingPipeline=os.path.join(os.path.dirname(__file__),"lpw_stable_diffusion.py")

logging.basicConfig(level=logging.DEBUG) #! Change to logging.INFO for less verbose logging


@click.command()
@click.option("-i", "--img", required=False, type=str)
@click.option("-b", "--batch", required=False, type=int, default=1)
@click.option("-p", "--prompt", required=False, type=str, default="")
@click.option("-n", "--neg", required=False, type=str, default="")
@click.option("-w", "--width", required=False, type=int, default=256)
@click.option("-h", "--height", required=False, type=int, default=256)
@click.option("-st", "--steps", required=False, type=int, default=28)
@click.option("-c", "--cfg", required=False, type=float, default=8)
@click.option("-s", "--seed", required=False, type=int, default=None)
def command_line(img,batch, prompt, neg, width, height, steps, cfg, seed):
    print("Please wait while the model is loaded...")
    pipe=try_load_model(model_path)
    while True:
        menu(img=img,batch=batch, prompt=prompt, neg=neg, width=width, height=height, steps=steps, cfg=cfg, seed=seed,pipe=pipe)

def check_output(name):
    if (os.path.exists("./Outputs/"+name)):
        if (name.count("_") == 0):
            name = name.split(".")[0] + "_1.png"
        else:
            name = name.split(".")[0].rsplit("_",1)[0] + "_" + str(int(name.split(".")[0].rsplit("_",1)[1]) + 1) + ".png"
        return check_output(name)
    else:
        return name

def try_load_model(model_path):
    try:
        model = DiffusionPipeline.from_pretrained(model_path,torch_dtype=torch.float32,use_safetensors=True,low_cpu_mem_usage=False,device_map=None,safety_checker=None,local_files_only=True,custom_pipeline=LongPromptWeightingPipeline)
        return model
    except RuntimeError:
        return None

def gen_txt2img(pipe,seed,width,height,steps,cfg,prompt,neg):
    print(f"\nUsing Seed  {seed}")
    pipe.to("cpu") 
    pipe.scheduler=DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,use_karras_sigmas=True)
    pipe.enable_vae_tiling()
    pipe.enable_vae_slicing()
    pipe.enable_attention_slicing("max")
    print(f"\nUsing Prompt  {prompt}")
    print(f"Using Negative Prompt  {neg}")
    result = pipe.text2img(prompt, height=height, width=width, num_inference_steps=steps, guidance_scale=cfg, negative_prompt=neg, generator=torch.Generator(device="cpu").manual_seed(seed))
    image = result.images[0]
    if (os.path.exists("./Outputs") == False):
        os.mkdir("./Outputs")
    name=check_output(f"{seed}.png")
    image.save("./Outputs/"+ name)
    print(f"Saved image to {name}")

def gen_img2img(pipe,seed,steps,cfg,img,prompt,neg):
    print(f"\nUsing Seed  {seed}")
    pipe.to("cpu")
    pipe.scheduler=DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,use_karras_sigmas=True)
    pipe.enable_vae_tiling()
    pipe.enable_vae_slicing()
    pipe.enable_attention_slicing("max")
    try:
        loaded_image = Image.open(img)
    except FileNotFoundError:
        print("Image not found")
        return
    result = pipe.img2img(prompt, image=loaded_image, num_inference_steps=steps, guidance_scale=cfg, negative_prompt=neg,seed=seed)
    image = result.images[0]
    if (os.path.exists("./Outputs") == False):
        os.mkdir("./Outputs")
    name=check_output(f"{seed}.png")
    image.save("./Outputs/"+ name)
    print(f"Saved image to {name}")
    
def print_time(start_time):
    if time.time() - start_time > 60:
        print(f"Time taken: {format((time.time() - start_time)/60,'.2f')}m")
    else:
        print(f"Time taken: {int(time.time() - start_time)}s")

def pre_txt2img(batch, prompt, neg, width, height, steps, cfg, seed,pipe):
    print("Starting txt2img...")
    if pipe==None:
        # throw error
        print("Failed to load model, pipe is NaN, aborting...")
        return
    for i in range(batch):
        print(f"Batch {i+1}/{batch}")
        starting_time = time.time()
        gen_txt2img(pipe=pipe,seed=seed,width=width,height=height,steps=steps,cfg=cfg,prompt=prompt,neg=neg)
        print_time(starting_time)
        seed+=1
        # we increment the seed or we randomize it?
        # seed= np.random.randint(np.iinfo(np.int32).max) 

def pre_img2img(img,batch, prompt, neg, width, height, steps, cfg, seed,pipe):
    print("Please enter the location of the image you want to use")
    while img is None:
        img = input()
        try :
            Image.open(img)
        except:
            print("Please enter a valid image/location")
            img = None
    print("Starting img2img...")

    if pipe==None:
        # throw error
        print("Failed to load model, pipe is NaN, aborting...")
        return
    
    for i in range(batch):
        loaded_image = Image.open(img)
        starting_time = time.time()
        gen_img2img(pipe=pipe,seed=seed,steps=steps,cfg=cfg,prompt=prompt,neg=neg,img=loaded_image)
        print_time(starting_time)
        seed+=1
        # we increment the seed or we randomize it?
        # seed= np.random.randint(np.iinfo(np.int32).max) 

def seed_menu(seed):
    while True:
        print(f"Would you like to use a seed? Current seed is {seed}")
        print("1. Yes")
        print("2. No, keep the current seed")
        print("3. Random")
        choice = input("Enter your choice: ")
        if choice == "1":
            seed = int(input("Enter your seed: ")) #sbagliato, non é unsigned 32 ma ce ne sbattiamo
            return seed
        elif choice == "2":
            if seed is None:
                seed = np.random.randint(np.iinfo(np.int32).max) if seed is None else seed
                print(f"Using a random seed of {seed}, since no seed was provided")
            print(f"Using seed {seed}")
            return seed
        elif choice == "3":
            seed = np.random.randint(np.iinfo(np.int32).max) if seed is None else seed
            print(f"Using a random seed of {seed}")
            return seed
        else:
            print("Invalid choice, please try again")

def prompt_menu(prompt):
    while True:
        print(f"Would you like to use a prompt? Current prompt is {prompt}")
        print("1. Yes")
        print("2. No, keep the current prompt")
        print("3. Random")
        choice = input("Enter your choice: ")
        if choice == "1":
            prompt = input("Enter your prompt: ")
            return prompt
        elif choice == "2":
            print(f"Using prompt {prompt}")
            return prompt
        elif choice == "3":
            prompt = str(rp.generate_prompt())
            print(f"Using a standard prompt of {prompt}")
            return prompt
        else:
            print("Invalid choice, please try again")

def neg_menu(neg):
    while True:
        print(f"Would you like to use a negative prompt? Current negative prompt is {neg}")
        print("1. Yes")
        print("2. No, keep the current negative prompt")
        print("3. Default")
        choice = input("Enter your choice: ")
        if choice == "1":
            neg = input("Enter your negative prompt: ")
            return neg
        elif choice == "2":
            print(f"Using negative prompt {neg}")
            return neg
        elif choice == "3":
            neg =NEG
            print(f"Using a standard negative prompt of {neg}")
            return neg
        else:
            print("Invalid choice, please try again")

def batch_menu(batch):
    while True:
        print(f"Would you like to use a batch size? Current batch size is {batch}")
        print("1. Yes")
        print("2. No, keep the current batch size")
        print("3. Default")
        choice = input("Enter your choice: ")
        if choice == "1":
            batch = int(input("Enter your batch size: "))
            return batch
        elif choice == "2":
            print(f"Using batch size {batch}")
            return batch
        elif choice == "3":
            batch = 1
            print(f"Using a standard batch size of {batch}")
            return batch
        else:
            print("Invalid choice, please try again")

def width_menu(width):
    while True:
        print(f"Would you like to use a width? [128 / 256 / 512] Current width is {width}")
        print("1. Yes")
        print("2. No, keep the current width")
        print("3. Default")
        choice = input("Enter your choice: ")
        if choice == "1":
            width = int(input("Enter your width: "))
            return width
        elif choice == "2":
            print(f"Using width {width}")
            return width
        elif choice == "3":
            width = WIDTH
            print(f"Using a standard width of {width}")
            return width
        else:
            print("Invalid choice, please try again")

def height_menu(height):
    while True:
        print(f"Would you like to use a height? [128 / 256 / 512] Current height is {height}")
        print("1. Yes")
        print("2. No, keep the current height")
        print("3. Default")
        choice = input("Enter your choice: ")
        if choice == "1":
            height = int(input("Enter your height: "))
            return height
        elif choice == "2":
            print(f"Using height {height}")
            return height
        elif choice == "3":
            height = HEIGHT
            print(f"Using a standard height of {height}")
            return height
        else:
            print("Invalid choice, please try again")

def steps_menu(steps):
    while True:
        print(f"Would you like to use a steps? Current steps is {steps}")
        print("1. Yes")
        print("2. No, keep the current steps")
        print("3. Default")
        choice = input("Enter your choice: ")
        if choice == "1":
            steps = input("Enter your steps: ")
            return steps
        elif choice == "2":
            print(f"Using steps {steps}")
            return steps
        elif choice == "3":
            steps = STEPS
            print(f"Using a standard steps of {steps}")
            return steps
        else:
            print("Invalid choice, please try again")

def cfg_menu(cfg):
    while True:
        print(f"Would you like to use a config? Current config is {cfg}")
        print("1. Yes")
        print("2. No, keep the current config")
        print("3. Default")
        choice = input("Enter your choice: ")
        if choice == "1":
            cfg = input("Enter your config: ")
            return cfg
        elif choice == "2":
            print(f"Using config {cfg}")
            return cfg
        elif choice == "3":
            cfg = CFG
            print(f"Using a standard config of {cfg}")
            return cfg
        else:
            print("Invalid choice, please try again")

def menu(img,batch, prompt, neg, width, height, steps, cfg, seed,pipe):
    print("Welcome to the Diffusion Models CLI")
    print("1. Generate an image from text")
    print("2. Generate an image from an image")
    print("3. Exit")
    choice = input("Enter your choice: ")
    if choice == "1":
        seed=seed_menu(seed)
        prompt=prompt_menu(prompt)
        neg=neg_menu(neg)
        pre_txt2img(batch=batch, prompt=prompt, neg=neg, width=width, height=height, steps=steps, cfg=cfg, seed=seed,pipe=pipe)
    elif choice == "2":
        seed=seed_menu(seed)
        prompt=prompt_menu(prompt)
        neg=neg_menu(neg)
        pre_img2img(img=img, batch=batch, prompt=prompt, neg=neg, width=width, height=height, steps=steps, cfg=cfg, seed=seed,pipe=pipe)    
    elif choice == "3":
        exit()
    else:
        print("Invalid choice, try again")
        menu(img=img, batch=batch, prompt=prompt, neg=neg, width=width, height=height, steps=steps, cfg=cfg, seed=seed,pipe=pipe)

def debug_menu(pipe):
    if pipe is None:
            pipe=try_load_model(model_path)
    print("Script running in command line mode but as app, going debug mode\n\n\n")
    print("Welcome to the Diffusion Models CLI Debug Mode")
    print("In this mode you can set all the parameters for the model")
    print("Would you like to use the default parameters?")
    print("1. Yes")
    print("2. No")
    choice = input("Enter your choice: ")
    if choice == "1":
        print("Ok, routing to default script...")
        while True:
            menu(img=None, batch=BATCH, prompt=PROMPT, neg=NEG, width=WIDTH, height=HEIGHT, steps=STEPS, cfg=CFG, seed=None,pipe=pipe)
    elif choice == "2":
        print("Ok, routing to custom...")
        seed=seed_menu(None)
        prompt=prompt_menu("")
        neg=neg_menu("")
        batch=batch_menu(BATCH)
        width=width_menu(WIDTH)
        height=height_menu(HEIGHT)
        steps=steps_menu(STEPS)
        cfg=cfg_menu(CFG)
        while True:
            print("1. Generate an image from text")
            print("2. Generate an image from an image")
            print("3. Back")
            choice = input("Enter your choice: ")
            if choice == "1":
                pre_txt2img(batch=batch, prompt=prompt, neg=neg, width=width, height=height, steps=steps, cfg=cfg, seed=seed,pipe=pipe)
            elif choice == "2":
                pre_img2img(img=None, batch=batch, prompt=prompt, neg=neg, width=width, height=height, steps=steps, cfg=cfg, seed=seed,pipe=pipe)
            elif choice == "3":
                debug_menu(pipe)
            else:
                print("Invalid choice, try again")
    elif choice == "3":
        print("You found the secret option, good job!\n Starting the secret script... Infinite loop incoming!\n\n\n")
        while True:
            pre_txt2img(batch=1, prompt=rp.generate_prompt(), neg=NEG, width=WIDTH, height=HEIGHT, steps=STEPS, cfg=CFG, seed=np.random.randint(np.iinfo(np.int32).max),pipe=pipe)
    else:
        print("Invalid choice, try again")
        debug_menu(pipe)

if __name__ == '__main__':
    debug=True
    if (sys.argv) == 1:
        debug_menu(pipe=None)
    else:
        command_line()