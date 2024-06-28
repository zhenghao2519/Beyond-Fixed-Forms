# beyond-fixed-forms
Query-Time Refinement for Open-Vocabulary 3D Instance Segmentation

### Installation
```
conda create -n BeyondFF python=3.8
conda activate BeyondFF
pip3 install torch==1.13.1+cu117 torchvision>=0.13.1+cu117 torchaudio>=0.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117 --no-cache-dir 
pip install -r requirements.txt
```



### Data preparation
Follows [Open3DIS/docs/DATA.md](https://github.com/VinAIResearch/Open3DIS/blob/main/docs/DATA.md)

Then use `./tools/ply2npy.py` transform .ply file under `./data/Scannet200/Scannet200_3D/val/original_ply_files` and store the transformed numpy file under `./data/Scannet200/Scannet200_3D/val/original_npy_files`.

### Run
Make sure all datapathes in `configs/config.yaml` are correct. Then run the following commands.
```
python tools/segmentation_2d.py --config configs/config.yaml

# or

python tools/segmentation_2d_yolo_world.py --config configs/config.yaml
```

```
python tools/projection_2d_to_3d.py --config configs/config.yaml
```

```
python tools/refinement.py --config configs/config.yaml
```

### Visualize
```
pip install -r requirements.txt
```

```
python visualization/visualize_scannet200.py  --config configs/config.yaml
```

### Evaluation
Manually set pathes in `evaluation/eval/eval_scannet200.py` and run
```
python evaluation/eval/eval_scannet200.py
```
and then check ./results.txt.