###############################################################################
# Check the classes to process in the main function before running this script
# Nvigate in Beyond-Fixed-Forms directory and run:
# python evaluation/eval/run_evl.py 2>&1 | tee script_log.txt

# While running, use following command in another terminal to check the output:
# less +F script_log.txt
# or:
# tail -f script_log.txt
###############################################################################
import subprocess
import os
from termcolor import colored
from tqdm import tqdm
import yaml
from munch import Munch
import sys
sys.path.append("./")
from evaluation.dataset.scannet200 import (
    BASE_CLASSES_SCANNET200,
    COMMON_CATS_SCANNET_200,
    HEAD_CATS_SCANNET_200,
    INSTANCE_CAT_SCANNET_200,
    NOVEL_CLASSES_SCANNET200,
    TAIL_CATS_SCANNET_200,
    VALID_CLASS_IDS_200_VALIDATION,
)
CHECKPOINT_FILE = "process_checkpoint.txt"
# CHECKPOINT_FILE = "./checkpoints/process_checkpoint.txt"

def run_command(command):
    try:
        result = subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command {' '.join(command)} failed with return code {e.returncode}")
        return False

def check_already_processed(file_path):
    return os.path.exists(file_path)


def read_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as file:
            checkpoint = yaml.safe_load(file)
            if checkpoint is None:
                return {}
            return checkpoint
    return {}

def write_checkpoint(checkpoint):
    with open(CHECKPOINT_FILE, 'w') as file:
        yaml.safe_dump(checkpoint, file)

def process_class(class_name, config_path, checkpoint):
    if checkpoint.get(class_name, {}).get("segmentation_2d", False):
        print(colored(f"Segmentation 2D for {class_name} already done, skipping.", "yellow"))
    else:
        if run_command(["python", "tools/segmentation_2d.py", "--config", config_path, "--cls", class_name]):
            checkpoint[class_name]["segmentation_2d"] = True
            write_checkpoint(checkpoint)
            print(colored(f"Segmentations 2D for {class_name} done.", "green"))
        else:
            return False

    # if checkpoint.get(class_name, {}).get("projection_2d_to_3d", False):
    #     print(colored(f"Projections 2D to 3D for {class_name} already done, skipping.", "yellow"))
    # else:
    #     if run_command(["python", "tools/projection_2d_to_3d.py", "--config", config_path, "--cls", class_name]):
    #         checkpoint[class_name]["projection_2d_to_3d"] = True
    #         write_checkpoint(checkpoint)
    #         print(colored(f"Projection 2D to 3D for {class_name} done.", "green"))
    #     else:
    #         return False

    # if checkpoint.get(class_name, {}).get("refinement", False):
    #     print(colored(f"Refinement for {class_name} already done, skipping.", "yellow"))
    # else:
    #     if run_command(["python", "tools/refinement.py", "--config", config_path, "--cls", class_name]):
    #         checkpoint[class_name]["refinement"] = True
    #         write_checkpoint(checkpoint)
    #         print(colored(f"Refinements for {class_name} done.", "green"))
    #     else:
    #         return False

    # if checkpoint.get(class_name, {}).get("evaluation", False):
    #     print(colored(f"Evaluation for {class_name} already done, skipping.", "yellow"))
    # else:
    #     if run_command(["python", "evaluation/eval/eval_scannet200.py", "--cls", class_name]):
    #         checkpoint[class_name]["evaluation"] = True
    #         write_checkpoint(checkpoint)
    #     else:
    #         return False

    return True

if __name__ == "__main__":
    config_path = "configs/config.yaml"
    cfg = Munch.fromDict(yaml.safe_load(open(config_path, "r").read()))
    checkpoint = read_checkpoint()

    classes_to_process = HEAD_CATS_SCANNET_200[:50] + COMMON_CATS_SCANNET_200[:50] + TAIL_CATS_SCANNET_200[:50]
    # nan_classes = ["clothes dryer", "pipe", "column", "bulletin board", "potted plant","coat rack", "folded chair", "calendar", "poster" ,"soap dispenser", "keyboard piano", "shower head", "guitar", "fire extinguisher", "tissue box", "scale", "soap dish", "tube", "plunger", "paper cutter", "storage container", "candle",  "music stand"   ]
    nan_classes = [  "bicycle", "machine",  "structure", "storage organizer", "potted plant", "cd case", "coat rack", "fire alarm", "power strip", "luggage"  ]
    # classes_to_process = ["clothes"]
    for class_name in tqdm(classes_to_process, desc="Processing classes"):
        
        if class_name in nan_classes:
            print("SKIP nan class")
            continue
        
        checkpoint.setdefault(class_name, {})
        print(colored(f"--------------Starting process for class: {class_name}--------------", "yellow", attrs=["bold"]))
        if process_class(class_name, config_path, checkpoint):
            print(colored(f"Class {class_name} processed successfully.", "green", attrs=["bold"]))
        else:
            print(colored(f"Class {class_name} processing failed.", "red", attrs=["bold"]))