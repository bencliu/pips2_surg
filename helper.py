import time
import numpy as np
import timeit
import saverloader
from nets.pips2 import Pips
import utils.improc
import utils.geom
import utils.misc
import random
from utils.basic import print_, print_stats
from datasets.exportdataset import ExportDataset
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from fire import Fire
import sys
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
from torch.utils.data import Dataset, DataLoader
import glob 
import os 
import json 
from pathlib import Path
import cv2 
import matplotlib.pyplot as plt

#External helper function 
def find_files_in_subdirectories(directory_path, extensions):
    found_files = []
    # Walk through the directory
    for root, _, files in os.walk(directory_path):
        # Check if the current directory is exactly two levels below the starting directory
        if root.count(os.sep) - directory_path.count(os.sep) == 1:
            for filename in files:
                if filename.endswith(extensions):
                    found_files.append(os.path.join(root, filename))
    
    return found_files

#External helper function 
def filter_data():
    print("Filtering") 
    dataset_folder = "/pasteur/u/bencliu/open_surg/data/surgical_hands_release/train_dataset2"
    npz_files = find_files_in_subdirectories(dataset_folder, ".npz") 
    print(len(npz_files))
    for file in npz_files:
        np_object = np.load(file, allow_pickle=True)
        d = dict(np_object)
        print(d['track_g'])

#External helper function 
def read_mp4(fn):
    vidcap = cv2.VideoCapture(fn)
    frames = []
    while(vidcap.isOpened()):
        ret, frame = vidcap.read()
        if ret == False:
            break
        frames.append(frame)
    vidcap.release()
    return frames

#External helper function 
def video_debugger():
    dataset_location = Path("/pasteur/u/bencliu/open_surg/data/surgical_hands_release/train_dataset")

    folder_names = dataset_location.glob('*/')
    folder_names = [fn for fn in folder_names]
    folder_names = sorted(list(folder_names))
    
    for folder in folder_names:
        rgbs = read_mp4(str(folder / 'rgb.mp4'))
        print(rgbs[0].shape)

#External helper function 
def process_frame_annot_helper(frame_annot, valid_frame, orig_dims):
    # Extract keypoints from the frame_annot dictionary
    default_height = 384 
    default_width = 512 
    orig_height, orig_width = orig_dims
    keypoints = frame_annot.get("keypoints", [])
    x_scale = default_width / orig_width
    y_scale = default_height / orig_height

    # Initialize an empty NumPy array to store the processed keypoints
    processed_keypoints = np.zeros((21, 4), dtype=float)

    # Iterate through the keypoints in groups of 3 (x, y, visible)
    for i in range(0, len(keypoints), 3):
        x, y, visible = keypoints[i:i + 3]
        new_x = int(x * x_scale)
        new_y = int(y * y_scale)
        
        # Convert valid_frame to 0 or 1
        if (x == 0 and y == 0) or not valid_frame:
            valid_frame_value = 0
        else:
            valid_frame_value = 1 
        visible = 1 if visible == 2 else 0  #n/a: 0, occ: 1, vis: 2
        
        # Populate the processed_keypoints array
        processed_keypoints[i // 3] = [new_x, new_y, visible, valid_frame_value]

    return processed_keypoints

#External helper function 
def video_helper(image_path_list, video_save_path, processed_annot=None, debug=True, frame_rate=30, index=0):
    default_height = 384 
    default_width = 512 
    debug_save_path = "/pasteur/u/bencliu/open_surg/data/surgical_hands_release/debug/image_" + str(index) + ".png" 

    # Get the dimensions of the first PNG image in the list
    img = cv2.imread(image_path_list[0], cv2.IMREAD_UNCHANGED)
    height, width = img.shape[:2]

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 video
    out = cv2.VideoWriter(video_save_path, fourcc, frame_rate, (default_width, default_height))

    # Loop through the PNG image files and add them to the video
    for i, image_path in enumerate(image_path_list):
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        try:
            img = cv2.resize(img, (default_width, default_height))
        except:
            print("skipping") 
            return 
        out.write(img)
    out.release()
    
    if debug:
        img = cv2.imread(image_path_list[0], cv2.IMREAD_UNCHANGED)
        resized_img = cv2.resize(img, (default_width, default_height))
        first_annot = processed_annot[0] 
        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB), aspect='auto')
        for x, y, _, _ in first_annot:
            plt.scatter(x, y, c='r', marker='o', s=20)
        
        # Save the resized image with keypoints
        plt.axis('off')
        plt.savefig(debug_save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

#External helper function 
def count_files_in_directory(directory_path):
    # Get a list of all files in the directory
    files = os.listdir(directory_path)
    
    # Filter the list to include only files (not directories)
    files = [file for file in files if os.path.isfile(os.path.join(directory_path, file))]
    
    # Return the number of files
    return len(files)


#Core function for processing surgical hands dataset 
def analyze_dataset():
    source_dir = "/pasteur/u/bencliu/open_surg/data/surgical_hands_release" 
    annot_path = "/pasteur/u/bencliu/open_surg/data/surgical_hands_release/annotations.json"
    image_root_path = "/pasteur/u/bencliu/open_surg/data/surgical_hands_release/images"
    save_dir = "/pasteur/u/bencliu/open_surg/data/surgical_hands_release/train_dataset3"
    
    SEQ_LEN = 30
    default_height = 384 
    default_width = 512 

    with open(annot_path, 'r') as json_file:
        ats = json.load(json_file)
    counter = 0 
    for k, v in ats.items():
        images = v['images'] 
        annot_i = v['annotations'] 
        first_index = 0 
        image_folder_path = os.path.join(image_root_path, images[0]['video_dir'])
        if not os.path.exists(image_folder_path):
            print(image_folder_path)
            continue 

        #Sort images and annotations  by image_id => TODO: Extact more frames by going over every single possible object from each video class 
        images = sorted(images, key=lambda x: x['id']) #Sorte by image_id 
        first_id = annot_i[0]['id'].split("_")[-1] #Default to first surgical hand 
        annot_i = [entry for entry in annot_i if entry.get('id').split("_")[-1] == first_id]
        annot_i = sorted(annot_i, key=lambda x: x['image_id'])

        #Limit params
        imageFileCount = count_files_in_directory(image_folder_path) 
        annotCount = len(annot_i) 
        print("Annot:", annotCount)

        while True: #Process each video 
            counter += 1 
            second_index = first_index + SEQ_LEN
            if second_index > imageFileCount or second_index > annotCount: #Ensure that annotations and images are matching together 
                break
            #Process images 
            subset_images = images[first_index:second_index]
            image_paths = sorted([os.path.join(image_root_path, x['video_dir'], x['file_name']) for x in subset_images])
            if not os.path.exists(image_paths[0]):
                break 
            valids = [x['is_labeled'] for x in subset_images] 

            #Get original dimensions 
            img = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED)
            orig_height, orig_width = img.shape[:2]

            #Process annotations 
            sub_annot = annot_i[first_index:second_index] #Attain annotations for image subset 
            processed_annot = [] #For all frames 
            for i, frame_annot in enumerate(sub_annot): #Each frame annotation item 
                frame_final_annot = process_frame_annot_helper(frame_annot, valids[i], (orig_height, orig_width))
                breakpoint()  #TODOO
                processed_annot.append(frame_final_annot)
            processed_annot_npz = np.array(processed_annot)

            #Sanity-checking annotation-fidelity:
            sanity_check(sub_annot[0], img, counter) 

            #Create video and save annotations 
            npz_dict = {"track_g" : processed_annot_npz}
            folder_path = os.path.join(save_dir, images[0]['video_dir'] + "_" + str(first_index))
            Path(folder_path).mkdir(parents=True, exist_ok=True)
            video_path = os.path.join(folder_path, "rgb.mp4") 
            annot_path = os.path.join(folder_path, "track.npz") 

            # -- save all files -- # 
            np.savez(annot_path, **npz_dict)
            video_helper(image_paths, video_path, processed_annot, debug=True, index=counter)

            first_index += 36 
            print("Completed processing video") 
    
#Helper function to plot surgical hands keypoint examples 
def sanity_check(annot, img, index):
    sub_lists = []
    for i in range(0, len(annot['keypoints']), 3):
        sub_lists.append(annot['keypoints'][i:i + 3])
    debug_save_path = "/pasteur/u/bencliu/open_surg/data/surgical_hands_release/debug/sanity_" + str(index) + ".png" 
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), aspect='auto')
    for x, y, _, in sub_lists:
        plt.scatter(x, y, c='r', marker='o', s=20)
    
    # Save the resized image with keypoints
    plt.axis('off')
    plt.savefig(debug_save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

if __name__ == "__main__":
    analyze_dataset() 