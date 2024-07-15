############################################
# Visualize result of a single class on a single scene. for quick tuning base on qualitative
# modify params, class and scene in singleviz_config.yaml
# python tools/singlevis_automation.py
############################################
import subprocess
import os
from termcolor import colored
from tqdm import tqdm
import yaml
from munch import Munch
import sys
sys.path.append("./")

def run_command(command):
    try:
        result = subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command {' '.join(command)} failed with return code {e.returncode}")
        return False


if __name__ == "__main__":
    config_path = "configs/sigleviz_config.yaml"
    cfg = Munch.fromDict(yaml.safe_load(open(config_path, "r").read()))
    # nan_classes = ["clothes dryer", "pipe", "column", "bulletin board", "potted plant","coat rack", "folded chair", "calendar", "poster" ,"soap dispenser", "keyboard piano", "shower head", "guitar", "fire extinguisher", "tissue box", "scale", "soap dish", "tube", "plunger", "paper cutter", "storage container", "candle",  "music stand"   ]
    class_name = cfg.base_prompt
    scene_id = cfg.scene_id

    # if run_command(["python", "tools/segmentation_2d_single.py", "--config", config_path, "--cls", class_name, "--scene", scene_id]):
    #     print(colored(f"Segmentations 2D for {class_name} done.", "green"))
    if True:

        # if run_command(["python", "tools/projection_2d_to_3d_single.py", "--config", config_path, "--cls", class_name]):
        #     print(colored(f"Projection 2D to 3D for {class_name} done.", "green"))

            if run_command(["python", "tools/refinement_single.py", "--config", config_path, "--cls", class_name]):
                print(colored(f"Refinements for {class_name} done.", "green"))

                if run_command(["python", "visualization/visualize_scannet200.py", "--config", config_path, "--cls", class_name, "--scene", scene_id]):
                    print(colored(f"Visualization for {class_name} generated.", "green"))
                    # run_command(["cd /home/jie_zhenghao/viz; python -m http.server 6008"])
    
    # uncomment if only vis:
    # run_command(["python", "visualization/visualize_scannet200.py", "--config", config_path, "--cls", class_name, "--scene", scene_id])
    



