# beyond-fixed-forms
Query-Time Refinement for Open-Vocabulary 3D Instance Segmentation

### Installation
```
pip install -r requirements.txt
```

### Data preparation
Follows [Open3DIS/docs/DATA.md](https://github.com/VinAIResearch/Open3DIS/blob/main/docs/DATA.md)

### Run
Make sure datapathes in `configs/config.yaml` are correct. Then run the following commands.
```
python tools/segmentation_2d.py --config configs/config.yaml
```

```
python tools/projection_2d_to_3d.py --config configs/config.yaml
```

### Visualize
```
pip install -r requirements.txt
```

```
python visualization/visualize_scannet200.py  --config configs/config.yaml
```