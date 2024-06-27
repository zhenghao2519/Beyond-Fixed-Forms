import torch
import numpy as np
import random
import os

import pyviz3d.visualizer as viz
import random
from os.path import join
import open3d as o3d

import argparse
from munch import Munch
import yaml

def generate_palette(n):
    palette = []
    for _ in range(n):
        red = random.randint(0, 255)
        green = random.randint(0, 255)
        blue = random.randint(0, 255)
        palette.append((red, green, blue))
    return palette

def rle_decode(rle):
    length = rle["length"]
    s = rle["counts"]

    starts, nums = [np.asarray(x, dtype=np.int32) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + nums
    mask = np.zeros(length, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask

def read_pointcloud(pcd_path):
    scene_pcd = o3d.io.read_point_cloud(str(pcd_path))
    print('READ pointcloud  '+str(scene_pcd))
    point = np.array(scene_pcd.points)
    color = np.array(scene_pcd.colors)

    return point, color

SCANNET200 = 'chair.table.door.couch.cabinet.shelf.desk.office_chair.bed.pillow.sink.picture.window.toilet.bookshelf.monitor.curtain.book.armchair.coffee_table.box.refrigerator.lamp.kitchen_cabinet.towel.clothes.tv.nightstand.counter.dresser.stool.cushion.plant.ceiling.bathtub.end_table.dining_table.keyboard.bag.backpack.toilet_paper.printer.tv_stand.whiteboard.blanket.shower_curtain.trash_can.closet.stairs.microwave.stove.shoe.computer_tower.bottle.bin.ottoman.bench.board.washing_machine.mirror.copier.basket.sofa_chair.file_cabinet.fan.laptop.shower.paper.person.paper_towel_dispenser.oven.blinds.rack.plate.blackboard.piano.suitcase.rail.radiator.recycling_bin.container.wardrobe.soap_dispenser.telephone.bucket.clock.stand.light.laundry_basket.pipe.clothes_dryer.guitar.toilet_paper_holder.seat.speaker.column.bicycle.ladder.bathroom_stall.shower_wall.cup.jacket.storage_bin.coffee_maker.dishwasher.paper_towel_roll.machine.mat.windowsill.bar.toaster.bulletin_board.ironing_board.fireplace.soap_dish.kitchen_counter.doorframe.toilet_paper_dispenser.mini_fridge.fire_extinguisher.ball.hat.shower_curtain_rod.water_cooler.paper_cutter.tray.shower_door.pillar.ledge.toaster_oven.mouse.toilet_seat_cover_dispenser.furniture.cart.storage_container.scale.tissue_box.light_switch.crate.power_outlet.decoration.sign.projector.closet_door.vacuum_cleaner.candle.plunger.stuffed_animal.headphones.dish_rack.broom.guitar_case.range_hood.dustpan.hair_dryer.water_bottle.handicap_bar.purse.vent.shower_floor.water_pitcher.mailbox.bowl.paper_bag.alarm_clock.music_stand.projector_screen.divider.laundry_detergent.bathroom_counter.object.bathroom_vanity.closet_wall.laundry_hamper.bathroom_stall_door.ceiling_light.trash_bin.dumbbell.stair_rail.tube.bathroom_cabinet.cd_case.closet_rod.coffee_kettle.structure.shower_head.keyboard_piano.case_of_water_bottles.coat_rack.storage_organizer.folded_chair.fire_alarm.power_strip.calendar.poster.potted_plant.luggage.mattress'
SCANNETV2 = 'cabinet.bed.chair.sofa.table.door.window.bookshelf.picture.counter.desk.curtain.refrigerator.shower_curtain.toilet.sink.bathtub'
class_names = SCANNET200.split('.')
CLASS_LABELS_200 = (
    "wall",
    "chair",
    "floor",
    "table",
    "door",
    "couch",
    "cabinet",
    "shelf",
    "desk",
    "office chair",
    "bed",
    "pillow",
    "sink",
    "picture",
    "window",
    "toilet",
    "bookshelf",
    "monitor",
    "curtain",
    "book",
    "armchair",
    "coffee table",
    "box",
    "refrigerator",
    "lamp",
    "kitchen cabinet",
    "towel",
    "clothes",
    "tv",
    "nightstand",
    "counter",
    "dresser",
    "stool",
    "cushion",
    "plant",
    "ceiling",
    "bathtub",
    "end table",
    "dining table",
    "keyboard",
    "bag",
    "backpack",
    "toilet paper",
    "printer",
    "tv stand",
    "whiteboard",
    "blanket",
    "shower curtain",
    "trash can",
    "closet",
    "stairs",
    "microwave",
    "stove",
    "shoe",
    "computer tower",
    "bottle",
    "bin",
    "ottoman",
    "bench",
    "board",
    "washing machine",
    "mirror",
    "copier",
    "basket",
    "sofa chair",
    "file cabinet",
    "fan",
    "laptop",
    "shower",
    "paper",
    "person",
    "paper towel dispenser",
    "oven",
    "blinds",
    "rack",
    "plate",
    "blackboard",
    "piano",
    "suitcase",
    "rail",
    "radiator",
    "recycling bin",
    "container",
    "wardrobe",
    "soap dispenser",
    "telephone",
    "bucket",
    "clock",
    "stand",
    "light",
    "laundry basket",
    "pipe",
    "clothes dryer",
    "guitar",
    "toilet paper holder",
    "seat",
    "speaker",
    "column",
    "bicycle",
    "ladder",
    "bathroom stall",
    "shower wall",
    "cup",
    "jacket",
    "storage bin",
    "coffee maker",
    "dishwasher",
    "paper towel roll",
    "machine",
    "mat",
    "windowsill",
    "bar",
    "toaster",
    "bulletin board",
    "ironing board",
    "fireplace",
    "soap dish",
    "kitchen counter",
    "doorframe",
    "toilet paper dispenser",
    "mini fridge",
    "fire extinguisher",
    "ball",
    "hat",
    "shower curtain rod",
    "water cooler",
    "paper cutter",
    "tray",
    "shower door",
    "pillar",
    "ledge",
    "toaster oven",
    "mouse",
    "toilet seat cover dispenser",
    "furniture",
    "cart",
    "storage container",
    "scale",
    "tissue box",
    "light switch",
    "crate",
    "power outlet",
    "decoration",
    "sign",
    "projector",
    "closet door",
    "vacuum cleaner",
    "candle",
    "plunger",
    "stuffed animal",
    "headphones",
    "dish rack",
    "broom",
    "guitar case",
    "range hood",
    "dustpan",
    "hair dryer",
    "water bottle",
    "handicap bar",
    "purse",
    "vent",
    "shower floor",
    "water pitcher",
    "mailbox",
    "bowl",
    "paper bag",
    "alarm clock",
    "music stand",
    "projector screen",
    "divider",
    "laundry detergent",
    "bathroom counter",
    "object",
    "bathroom vanity",
    "closet wall",
    "laundry hamper",
    "bathroom stall door",
    "ceiling light",
    "trash bin",
    "dumbbell",
    "stair rail",
    "tube",
    "bathroom cabinet",
    "cd case",
    "closet rod",
    "coffee kettle",
    "structure",
    "shower head",
    "keyboard piano",
    "case of water bottles",
    "coat rack",
    "storage organizer",
    "folded chair",
    "fire alarm",
    "power strip",
    "calendar",
    "poster",
    "potted plant",
    "luggage",
    "mattress",
)
BENCHMARK_SEMANTIC_IDXS = [
        1,
        3,
        2,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        21,
        22,
        23,
        24,
        26,
        27,
        28,
        29,
        31,
        32,
        33,
        34,
        35,
        36,
        38,
        39,
        40,
        41,
        42,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        54,
        55,
        56,
        57,
        58,
        59,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        82,
        84,
        86,
        87,
        88,
        89,
        90,
        93,
        95,
        96,
        97,
        98,
        99,
        100,
        101,
        102,
        103,
        104,
        105,
        106,
        107,
        110,
        112,
        115,
        116,
        118,
        120,
        121,
        122,
        125,
        128,
        130,
        131,
        132,
        134,
        136,
        138,
        139,
        140,
        141,
        145,
        148,
        154,
        155,
        156,
        157,
        159,
        161,
        163,
        165,
        166,
        168,
        169,
        170,
        177,
        180,
        185,
        188,
        191,
        193,
        195,
        202,
        208,
        213,
        214,
        221,
        229,
        230,
        232,
        233,
        242,
        250,
        261,
        264,
        276,
        283,
        286,
        300,
        304,
        312,
        323,
        325,
        331,
        342,
        356,
        370,
        392,
        395,
        399,
        408,
        417,
        488,
        540,
        562,
        570,
        572,
        581,
        609,
        748,
        776,
        1156,
        1163,
        1164,
        1165,
        1166,
        1167,
        1168,
        1169,
        1170,
        1171,
        1172,
        1173,
        1174,
        1175,
        1176,
        1178,
        1179,
        1180,
        1181,
        1182,
        1183,
        1184,
        1185,
        1186,
        1187,
        1188,
        1189,
        1190,
        1191,
    ]

class VisualizationScannet200:
    def __init__(self, point, color):
        self.point = point
        self.color = color
        self.vis = viz.Visualizer()
        self.vis.add_points(f'pcl', point, color.astype(np.float32), point_size=20, visible=True)
    
    def save(self, path):
        self.vis.save(path)
    
    def superpointviz(self, spp_path):
        print('...Visualizing Superpoints...')
        spp = torch.from_numpy(torch.load(spp_path)).to(device='cuda')
        unique_spp, spp, num_point = torch.unique(spp, return_inverse=True, return_counts=True)
        n_spp = unique_spp.shape[0]
        pallete =  generate_palette(n_spp + 1)
        uniqueness = torch.unique(spp).clone()
        # skip -1 
        tt_col = self.color.copy()
        for i in range(0, uniqueness.shape[0]):
            ss = torch.where(spp == uniqueness[i].item())[0]
            for ind in ss:
                tt_col[ind,:] = pallete[int(uniqueness[i].item())]
        self.vis.add_points(f'superpoint: ' + str(i), self.point, tt_col, point_size=20, visible=True)
        print('---Done---')
    
    def gtviz(self, gt_data, specific = False):
        print('...Visualizing Groundtruth...')
        normalized_point, normalized_color, sem_label, ins_label = torch.load(gt_data)
        print ("DEBUG type of values in ins_label: ", ins_label.dtype)
        sem_label = sem_label.astype(np.int32)
        ins_label = ins_label.astype(np.int32)

        pallete =  generate_palette(int(2e3 + 1))
        n_label = np.unique(ins_label)
        print("DEBUG: ", np.unique(sem_label))
        tt_col = self.color.copy()
        for i in range(0, n_label.shape[0]):
            if sem_label[np.where(ins_label==n_label[i])][0] == 0 or sem_label[np.where(ins_label==n_label[i])][0] == 1: # Ignore wall/floor
                continue
            tt_col[np.where(ins_label==n_label[i])] = pallete[i]
            if specific: # be more specific
                print("Running ins mask", i, "with label", [sem_label[np.where(ins_label==n_label[i])][0]], CLASS_LABELS_200[ BENCHMARK_SEMANTIC_IDXS.index(sem_label[np.where(ins_label==n_label[i])][0])])
                tt_col_specific = self.color.copy()
                tt_col_specific[np.where(ins_label==n_label[i])] = pallete[i]
                self.vis.add_points(f'GT instance: ' + str(i) + '_' + CLASS_LABELS_200[ BENCHMARK_SEMANTIC_IDXS.index(sem_label[np.where(ins_label==n_label[i])][0])], self.point, tt_col_specific, point_size=20, visible=True)

        self.vis.add_points(f'GT instance: ' + str(i), self.point, tt_col, point_size=20, visible=True)
        print('---Done---')

    def vizmask3d(self, mask3d_path, specific = False):
        print('...Visualizing 3D backbone mask...')
        dic = torch.load(mask3d_path)
        instance = dic['ins']
        conf3d = dic['conf']
        pallete =  generate_palette(int(2e3 + 1))
        tt_col = self.color.copy()
        limit = 10
        for i in range(0, instance.shape[0]):
            tt_col[instance[i] == 1] = pallete[i]
            if specific and limit > 0: # be more specific but limit 10 masks (avoiding lag)
                limit -= 1
                tt_col_specific = self.color.copy()
                tt_col_specific[instance[i] == 1] = pallete[i]
                self.vis.add_points(f'3D backbone mask: ' + str(i) + '_' + str(conf3d[i]), self.point, tt_col_specific, point_size=20, visible=True)

        self.vis.add_points(f'3D backbone mask: ' + str(i), self.point, tt_col, point_size=20, visible=True)
        print('---Done---')

    def vizmask2d(self, mask2d_path, specific = False):
        print('...Visualizing 2D lifted mask...')
        dic = torch.load(mask2d_path)
        instance = dic['ins']
        instance = torch.stack([torch.tensor(rle_decode(ins)) for ins in instance])
        conf2d = dic['conf'] # confidence really doesn't affect much (large mask -> small conf)
        pallete =  generate_palette(int(5e3 + 1))
        tt_col = self.color.copy()
        limit = 10
        for i in range(0, instance.shape[0]):
            tt_col[instance[i] == 1] = pallete[i]
            if specific and limit > 0: # be more specific but limit 10 masks (avoiding lag)
                limit -= 1
                tt_col_specific = self.color.copy()
                tt_col_specific[instance[i] == 1] = pallete[i]
                self.vis.add_points(f'2D lifted mask: ' + str(i) + '_' + str(conf2d[i].item())[:5], self.point, tt_col_specific, point_size=20, visible=True)

        self.vis.add_points(f'2D lifted mask: ' + str(i), self.point, tt_col, point_size=20, visible=True)
        print('---Done---')        
        
    def finalviz(self, agnostic_path, specific = False, vocab = False):
        print('...Visualizing final class agnostic mask...')
        dic = torch.load(agnostic_path)
        # print(dic.keys(), dic['conf'], dic['final_class'])
        instance = dic['ins']
        # print("TEST" + str(instance[0]))
        instance = torch.stack([torch.tensor(rle_decode(ins)) for ins in instance])
        conf2d = dic['conf'] # confidence really doesn't affect much (large mask -> small conf)

        if vocab == True:
            label = dic['final_class']
        pallete =  generate_palette(int(2e3 + 1))
        tt_col = self.color.copy()
        limit = 50
        for i in range(0, instance.shape[0]):
            # print('DEBUG   '+str(instance.shape) + str(len(pallete)) + str(len(tt_col)))
            tt_col[instance[i] == 1] = pallete[i]
            if specific and limit > 0: # be more specific but limit 10 masks (avoiding lag)
                limit -= 1
                tt_col_specific = self.color.copy()
                tt_col_specific[instance[i] == 1] = pallete[i]
                if vocab == True:
                    self.vis.add_points(f'final mask: ' + str(i) + '_' + class_names[label[i]], self.point, tt_col_specific, point_size=20, visible=True)                
                else:
                    self.vis.add_points(f'final mask: ' + str(i) + '_' + str(conf2d[i].item())[:5], self.point, tt_col_specific, point_size=20, visible=True)

        self.vis.add_points(f'final mask: ' + str(i), self.point, tt_col, point_size=20, visible=True)
        print('---Done---')  


    def singleviz(self, agnostic_path, specific = False, vocab = False):
            print('...Visualizing single class agnostic mask...')
            dic = torch.load(agnostic_path, map_location='cpu')
            # print(dic.keys(), dic['conf'], dic['final_class'])
            instance = dic['ins'].cpu()
            print(instance.get_device())
            # print("TEST" + str(instance[0]))
            # instance = torch.stack([torch.tensor(rle_decode(ins)) for ins in instance])
            conf2d = dic['conf'] # confidence really doesn't affect much (large mask -> small conf)

            if vocab == True:
                label = dic['final_class']
            pallete =  generate_palette(int(2e3 + 1))
            tt_col = self.color.copy()
            limit = 40
            print("instance shape", instance.shape)
            for i in range(0, instance.shape[0]):
                # print('DEBUG   '+str(instance.shape) + str(len(pallete)) + str(len(tt_col)))
                tt_col[instance[i] == 1] = pallete[i]
                if specific and limit > 0: # be more specific but limit 10 masks (avoiding lag)
                    limit -= 1
                    tt_col_specific = self.color.copy()
                    tt_col_specific[instance[i] == 1] = pallete[i]
                    if vocab == True:
                        self.vis.add_points(f'single object mask: ' + str(i) + '_' + label[i] + '_' + str(conf2d[i].item())[:5], self.point, tt_col_specific, point_size=20, visible=True)                
                    else:
                        self.vis.add_points(f'single object mask: ' + str(i) + '_' + str(conf2d[i].item())[:5], self.point, tt_col_specific, point_size=20, visible=True)

            self.vis.add_points(f'single object mask: ' + str(i), self.point, tt_col, point_size=20, visible=True)
            print('---Done---')  
            
    def refinedviz(self, agnostic_path, specific = False, vocab = False):
        print('...Visualizing refined class agnostic mask...')
        dic = torch.load(agnostic_path, map_location='cpu')
        # print(dic.keys(), dic['conf'], dic['final_class'])
        instance = dic['ins'].cpu()
        print(instance.get_device())
        # print("TEST" + str(instance[0]))
        # instance = torch.stack([torch.tensor(rle_decode(ins)) for ins in instance])
        conf2d = dic['conf'] # confidence really doesn't affect much (large mask -> small conf)

        if vocab == True:
            label = dic['final_class']
        pallete =  generate_palette(int(2e3 + 1))
        tt_col = self.color.copy()
        limit = 20
        print("instance shape", instance.shape)
        for i in range(0, instance.shape[0]):
            # print('DEBUG   '+str(instance.shape) + str(len(pallete)) + str(len(tt_col)))
            tt_col[instance[i] == 1] = pallete[i]
            if specific and limit > 0: # be more specific but limit 10 masks (avoiding lag)
                limit -= 1
                tt_col_specific = self.color.copy()
                tt_col_specific[instance[i] == 1] = pallete[i]
                if vocab == True:
                    self.vis.add_points(f'refined object mask: ' + str(i) + '_' + label[i] + '_' + str(conf2d[i].item())[:5], self.point, tt_col_specific, point_size=20, visible=True)                
                else:
                    self.vis.add_points(f'refined object mask: ' + str(i) + '_' + str(conf2d[i].item())[:5], self.point, tt_col_specific, point_size=20, visible=True)

        self.vis.add_points(f'refined object mask: ' + str(i), self.point, tt_col, point_size=20, visible=True)
        print('---Done---')  


def get_parser():
    parser = argparse.ArgumentParser(description="Configuration Open3DIS")
    parser.add_argument("--config",type=str,required = True,help="Config")
    return parser


if __name__ == "__main__":
    
    '''
        Visualization using PyViz3D
        1. superpoint visualization
        2. ground-truth annotation
        3. 3D backbone mask (isbnet, mask3d) -- class-agnostic
        4. lifted 2D masks -- class-agnostic
        5. final masks --class-agnostic (2D+3D)
        
    
    '''
    args = get_parser().parse_args()
    cfg = Munch.fromDict(yaml.safe_load(open(args.config, "r").read()))
    
    # Scene ID to visualize
    scene_id = 'scene0435_00'

    ##### The format follows the dataset tree
    ## 1
    check_superpointviz = False
    spp_path = './data/Scannet200/Scannet200_3D/val/superpoints/' + scene_id + '.pth'
    ## 2
    check_gtviz = True
    gt_path = './data/Scannet200/Scannet200_3D/val/groundtruth/' + scene_id + '.pth'
    ## 3
    check_3dviz = False
    mask3d_path = './data/Scannet200/Scannet200_3D/val/isbnet_clsagnostic_scannet200/' + scene_id + '.pth'
    ## 4
    check_2dviz = False
    mask2d_path = '../exp/version_sam/hier_agglo/' + scene_id + '.pth'
    ## 5 Visualize final mask of stage 1
    check_finalviz = False
    agnostic_path = os.path.join(cfg.stage_1_result_dir, scene_id + '.pth')
    ## 6 Visualize final mask of stage 2 in each class
    check_singleviz = True
    output_dir = os.path.join(cfg.mask_3d_dir, cfg.base_prompt)
    ## 7 Visualize final refined masks
    check_refinedviz = True
    refined_path = os.path.join(cfg.final_output_dir, cfg.base_prompt, scene_id + '.pth')
    # agnostic_path = './data/Scannet200/Scannet200_3D/val/single_object_test/' + scene_id + '.pth'

    pyviz3d_dir = '../viz' # visualization directory

    # Visualize Point Cloud 
    ply_file = './data/Scannet200/Scannet200_3D/original_ply_files'
    point, color = read_pointcloud(os.path.join(ply_file,scene_id + '.ply'))
    color = color * 127.5
    print(len(color))

    VIZ = VisualizationScannet200(point, color)    
    
    if check_superpointviz:
        VIZ.superpointviz(spp_path)
    if check_gtviz:
        VIZ.gtviz(gt_path, specific = True)
    if check_3dviz:
        VIZ.vizmask3d(mask3d_path, specific = False)
    if check_2dviz:
        VIZ.vizmask2d(mask2d_path, specific = False)
    if check_finalviz:
        VIZ.finalviz(agnostic_path, specific = True, vocab = True)
    if check_singleviz:
        VIZ.singleviz(os.path.join(output_dir,scene_id + '.pth'), specific = True, vocab = False)
    if check_refinedviz:
        VIZ.refinedviz(refined_path, specific = True, vocab = True)
    VIZ.save(pyviz3d_dir)
