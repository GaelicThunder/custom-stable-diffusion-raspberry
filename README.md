# Script for loading Custom Stable Diffusion models for Raspberry Pi, with DPM++ 2M Karras like and prompt weight like the WebUI

This repository contains scripts to load and generate custom stable diffusion models, optimized for a Raspberry Pi 4 with 4GB or more memory (tested on a 4Gb, Swap memory needed). It's nothing fast, 20 min for a 256x256, but you can do some fancy stuff like this:

![Epaper Screen](https://github.com/GaelicThunder/custom-stable-diffusion-raspberry/blob/main/img/epaper.jpg)

The scripts facilitate the use of `.safetensors` and `.ckpt` files (such as those from Civitai) with a command line or a Raspberry Pi, since the informations for loading such models always refers to the most used tools. Before these can be used, they must be converted to a stable diffusion model using the `convert_original_stable_diffusion_to_diffusers.py` script from the HuggingFace `diffusers` repository.
The main script uses the DPMSolverMultistepScheduler which is very similar to the DPM++ 2M Karras from Automatic1111 webui.
It also uses lpw_stable_diffusion.py which load the custom pipeline, this bad guy lets us use the long prompt weighting like in the Web UI.
The main goal is to have a very similar generation like the one you have with the webUI without using the webUI, which is VERY complicate to do a raspi.

## Getting Started

1. Clone this repository to your local machine.
2. Create a virtual env and then install the required Python(use 10, 11 wasn't working) libraries using the provided `requirements.txt` file. This can be done by running the following command in your terminal:
    ```
    pip install virtualenv
    python3.10 -m venv env
    pip install -r requirements.txt
    ```
3. Clone the `diffusers` repository from HuggingFace. You can do this with the following command:
    ```
    git clone https://github.com/huggingface/diffusers
    ```
4. Convert your model to a stable diffusion model using the following command:
    ```
    python diffusers/scripts/convert_original_stable_diffusion_to_diffusers.py --checkpoint_path ur_model_location.safetensors --extract_ema --dump_path Where_to_save_it --from_safetensors
    ```
    Replace `ur_model_location.safetensors` with the path to your model and `Where_to_save_it` with the location where you want to save the converted model.
    Example:
    ```
    python diffusers/scripts/convert_original_stable_diffusion_to_diffusers.py --checkpoint_path SD1_5.safetensors --extract_ema --dump_path SD1_5 --from_safetensors
    ```
    > at this point a folder called SD1_5 will be created

5. In the `custom_stable_diff.py` script in this repository, update the `model_path` variable to point to your local model folder. The line to change is as follows:
    ```
    model_path = os.path.join(os.path.dirname(__file__),"my_local_model_folder")
    ```
    Replace `"my_local_model_folder"` with the name of your model's folder.
    Example:
    if we have the SD1_5 folder:
    ```
    model_path = os.path.join(os.path.dirname(__file__),"SD1_5")
    ```

## Scripts

- `custom_stable_diff.py`: This is the main script that loads and generates txt2img or img2img. Accepts optional arguments

- `lpw_stable_diffusion.py`: [Load the custom pipeline, this bad guy lets us use the long prompt weighting like in the Web UI.]

Please note that more detailed usage instructions and options for each script are included in comments within the scripts themselves.

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

