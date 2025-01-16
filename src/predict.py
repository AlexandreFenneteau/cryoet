import os
import glob
import numpy as np
from collections import Counter

from ultralytics import YOLO
import networkx as nx
import pandas as pd


def iou(box_1, box_2):  
    # - box_1: A dictionary with keys 'x1', 'y1', 'x2', 'y2' representing the first box.  
    # - box_2: A dictionary with keys 'x1', 'y1', 'x2', 'y2' representing the second box.  
  
    assert box_1['x1'] < box_1['x2'], "Invalid box_1 coordinates"  
    assert box_1['y1'] < box_1['y2'], "Invalid box_1 coordinates"  
    assert box_2['x1'] < box_2['x2'], "Invalid box_2 coordinates"  
    assert box_2['y1'] < box_2['y2'], "Invalid box_2 coordinates"  
  
    # Calculate the intersection box  
    x1_inter = max(box_1['x1'], box_2['x1'])  
    y1_inter = max(box_1['y1'], box_2['y1'])  
    x2_inter = min(box_1['x2'], box_2['x2'])  
    y2_inter = min(box_1['y2'], box_2['y2'])  
  
    # Compute the intersection area  
    inter_width = max(0, x2_inter - x1_inter)  
    inter_height = max(0, y2_inter - y1_inter)  
    inter_area = inter_width * inter_height  
  
    if inter_area == 0.:  
        return 0.  
    else:  
        # Compute the areas of box_1 and box_2  
        box_1_area = (box_1['x2'] - box_1['x1']) * (box_1['y2'] - box_1['y1'])  
        box_2_area = (box_2['x2'] - box_2['x1']) * (box_2['y2'] - box_2['y1'])  
  
        # Compute the union area  
        union_area = box_1_area + box_2_area - inter_area  
  
        # Compute the IoU  
        iou = inter_area / union_area if union_area != 0. else 0.  
        return iou  


def compatible_cls(cls_1, cls_2):
    if cls_1 == cls_2:
        return True
    else:
        diam_particles = [60, 90, 150, 130, 135]
        diam_1 = diam_particles[cls_1]
        diam_2 = diam_particles[cls_2]
        if max(diam_1, diam_2) > 1.2 * min(diam_1, diam_2):
            return False
        else:
            return True

def get_majority(cls_list):
    most_commons = Counter(cls_list).most_common(1)
    chosen_common = most_commons[-1] #in case of tie, we take the last one
    major_cls = chosen_common[0]
    return major_cls


def filter_particles(isol_particles, graph,
                     particle_min_slices={0: 2, 1: 2, 2: 2, 3: 3, 4: 3},
                     sum_confidence_thresh={0: 0.1, 1: 0.1, 2: 0.2, 3: 0.3, 4: 0.3}):
    filtered_particles = []
    for isol_part in isol_particles:
        node_clss = []
        sum_conf = 0.
        for node_id in isol_part:
            node_vals = graph.nodes[node_id]
            node_clss.append(node_vals['c'])
            sum_conf += node_vals['conf']
        majority_clss = get_majority(node_clss)
        if len(isol_part) >= particle_min_slices[majority_clss] and sum_conf >= sum_confidence_thresh[majority_clss]:
            filtered_particles.append(isol_part)
    return filtered_particles


def create_bbox_graph(yolo_results, z_max):
    g = nx.Graph()

    i = 0
    for z, result in enumerate(yolo_results):
        boxes = result.boxes  # Boxes object for bounding box outputs
        for box in boxes:
            xyxyn_tensor = box.xyxyn.squeeze(0)
            g.add_node(i,
                       x1=float(xyxyn_tensor[0]),
                       y1=float(xyxyn_tensor[1]),
                       x2=float(xyxyn_tensor[2]),
                       y2=float(xyxyn_tensor[3]),
                       z=z/z_max,
                       c=int(box.cls.squeeze(0)),
                       conf=float(box.conf.squeeze(0))
                       )
            i+= 1
    return g


def create_graph_edges(g, z_max, consecutive_authorized_slices=6):
    consecutive_depth_limit = consecutive_authorized_slices / z_max
    for node_id in g.nodes:
        node_vals = g.nodes[node_id]
        for other_node_id in range(node_id+1, len(g.nodes)):
            other_node_vals = g.nodes[other_node_id]
            if other_node_vals['z'] - node_vals['z'] < consecutive_depth_limit:
                if compatible_cls(node_vals['c'], other_node_vals['c']):
                    weight = iou(node_vals, other_node_vals)
                    if weight > 0.:
                        g.add_edge(node_id, other_node_id, weight=weight)
    return g


def particles_to_dicts(graph, particle_nodes, 
                    exp_name,
                    img_size = {'x': 630, 'y': 630, 'z': 184}, 
                    res={"z": 10.012444196428572, "y": 10.012444196428572, "x": 10.012444537618887}):
    cls_dict = {0: "apo-ferritin", 1: "beta-galactosidase",
                2: "ribosome", 3: "thyroglobulin", 4: "virus-like-particle"}
    data_dicts = []
    for part in particle_nodes:
        node_clss = []
        confs = []
        x_centers = []
        y_centers = []
        zs = []
        for node_id in part:
            node_vals = graph.nodes[node_id]
            node_clss.append(node_vals['c'])
            confs.append(node_vals['conf'])
            x_centers.append((node_vals['x2'] + node_vals['x1'])/2.)
            y_centers.append((node_vals['y2'] + node_vals['y1'])/2.)
            zs.append(node_vals['z'])
        majority_clss = get_majority(node_clss)
        x_center = np.average(x_centers, weights=confs) * (img_size['x'] - 1.) * res['x']
        y_center = np.average(y_centers, weights=confs) * (img_size['y'] - 1.) * res['y']
        z_center = np.average(zs, weights=confs) * (img_size['z'] - 1.) * res['z']
        data_dicts.append({"experiment": exp_name,
                           "particle_type": cls_dict[majority_clss],
                           "x": x_center,
                           "y": y_center,
                           "z": z_center})
    return data_dicts
    

if __name__ == "__main__":

    test_exams = ["TS_6_6"]
    # test_exams = ["TS_6_6", "TS_73_6"]
    test_img_dir = r"C:\Users\AlexandreFenneteau\Travail\perso\cryoet\data\preproc\yolo\axial\images\test"
    model_path = r"C:\Users\AlexandreFenneteau\Travail\perso\cryoet\data\training\kaggle_train_axial_v3\weights\best.pt"

    model = YOLO(model_path)

    particle_dicts = []
    for exam in test_exams:
        slices = sorted(glob.glob(os.path.join(test_img_dir, f"{exam}_*.png")))
        n_slices = len(slices)
        z_max = n_slices - 1
        
        results = model(slices)
        exam_graph = create_bbox_graph(results, z_max)
        exam_graph = create_graph_edges(exam_graph, z_max, 6)
        graph_communities = nx.community.louvain_communities(exam_graph)
        filtered_particles = filter_particles(graph_communities, exam_graph)
        particle_dicts = particle_dicts + particles_to_dicts(exam_graph, filtered_particles, exam)
    df = pd.DataFrame(particle_dicts)