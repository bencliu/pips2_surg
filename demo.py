import time
import numpy as np
import saverloader
from nets.pips2 import Pips
import utils.improc
from utils.basic import print_, print_stats
import torch
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from fire import Fire
import sys
import cv2
from pathlib import Path
import json 
import os 
import random
import math

def debug():
    annot_path = "/pasteur/u/bencliu/open_surg/data/avos/hand_keypoints_train_boot_aug_2.json"
    mp4_path = "/pasteur/data/AVOS/surgery-multitask-videos-and-cache"
    with open(annot_path, 'r') as json_file:
        data_dict = json.load(json_file)
        breakpoint() 



#Helper function: Video => List of Frames 
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

def run_model(model, rgbs, S_max=128, N=64, iters=16, sw=None, custom_points=None):
    rgbs = rgbs.cuda().float() # B, S, C, H, W

    B, S, C, H, W = rgbs.shape
    assert(B==1)

    # pick N points to track; we'll use a uniform grid
    N_ = np.sqrt(N).round().astype(np.int32)
    grid_y, grid_x = utils.basic.meshgrid2d(B, N_, N_, stack=False, norm=False, device='cuda')
    grid_y = 8 + grid_y.reshape(B, -1)/float(N_-1) * (H-16)
    grid_x = 8 + grid_x.reshape(B, -1)/float(N_-1) * (W-16)
    xy0 = torch.stack([grid_x, grid_y], dim=-1) # B, N_*N_, 2
    _, S, C, H, W = rgbs.shape

    #TODO 
    if custom_points:
        trajs_e = torch.tensor(custom_points, device='cuda').repeat(1,S,1,1)
    else:
        trajs_e = xy0.unsqueeze(1).repeat(1,S,1,1)
        breakpoint() 

    #TODO - ensure that custom_points shape match the original trajs_e 

    iter_start_time = time.time()
    
    preds, preds_anim, _, _ = model(trajs_e, rgbs, iters=iters, feat_init=None, delta_mult=0.5)
    trajs_e = preds[-1]

    iter_time = time.time()-iter_start_time
    print('inference time: %.2f seconds (%.1f fps)' % (iter_time, S/iter_time))

    if sw is not None and sw.save_this:
        rgbs_prep = utils.improc.preprocess_color(rgbs)
        sw.summ_traj2ds_on_rgbs('outputs/trajs_on_rgbs', trajs_e[0:1], utils.improc.preprocess_color(rgbs[0:1]), cmap='hot', linewidth=1, show_dots=False)
    return trajs_e


def main(
        filename='./stock_videos/camel.mp4',
        S=48, # seqlen
        N=1024, # number of points per clip
        stride=8, # spatial stride of the model
        timestride=1, # temporal stride of the model
        iters=16, # inference steps of the model
        image_size=(512,896), # input resolution
        max_iters=30, # number of clips to run
        shuffle=False, # dataset shuffling
        log_freq=1, # how often to make image summaries
        log_dir='./logs_demo',
        init_dir='./reference_model',
        device_ids=[0],
        custom_points=None 
):

    # the idea in this file is to run the model on a demo video,
    # and return some visualizations
    
    exp_name = 'de00' # copy from dev repo

    print('filename', filename)
    name = Path(filename).stem
    print('name', name)
    
    rgbs = read_mp4(filename)
    rgbs = np.stack(rgbs, axis=0) # S,H,W,3
    rgbs = rgbs[:,:,:,::-1].copy() # BGR->RGB
    rgbs = rgbs[::timestride]
    S_here,H,W,C = rgbs.shape
    print('rgbs', rgbs.shape)

    # autogen a name
    model_name = "%s_%d_%d_%s" % (name, S, N, exp_name)
    import datetime
    model_date = datetime.datetime.now().strftime('%H:%M:%S')
    model_name = model_name + '_' + model_date
    print('model_name', model_name)
    
    writer_t = SummaryWriter(log_dir + '/' + model_name + '/t', max_queue=10, flush_secs=60)

    global_step = 0

    model = Pips(stride=8).cuda()
    parameters = list(model.parameters())
    if init_dir:
        _ = saverloader.load(init_dir, model)
    global_step = 0
    model.eval()

    idx = list(range(0, max(S_here-S,1), S))
    if max_iters:
        idx = idx[:max_iters]
    
    for si in idx:
        global_step += 1
        
        iter_start_time = time.time()

        sw_t = utils.improc.Summ_writer(
            writer=writer_t,
            global_step=global_step,
            log_freq=log_freq,
            fps=8,
            scalar_freq=int(log_freq/2),
            just_gif=True)

        rgb_seq = rgbs[si:si+S]
        rgb_seq = torch.from_numpy(rgb_seq).permute(0,3,1,2).to(torch.float32) # S,3,H,W
        rgb_seq = F.interpolate(rgb_seq, image_size, mode='bilinear').unsqueeze(0) # 1,S,3,H,W
        
        with torch.no_grad():
            trajs_e = run_model(model, rgb_seq, S_max=S, N=N, iters=iters, sw=sw_t, custom_points=custom_points)

        iter_time = time.time()-iter_start_time
        
        print('%s; step %06d/%d; itime %.2f' % (
            model_name, global_step, max_iters, iter_time))
        
            
    writer_t.close()



def create_mp4_from_images(image_dir_path, output_file_path, fps=30):
    # Get the list of image files in the directory
    image_files = [f for f in os.listdir(image_dir_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Sort the image files to maintain the correct order
    image_files.sort()
    
    # Get the dimensions of the first image to determine video size
    first_image_path = os.path.join(image_dir_path, image_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape
    
    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec for MP4 format
    out = cv2.VideoWriter(output_file_path, fourcc, fps, (width, height))
    
    # Loop through image files and add them to the video
    for image_file in image_files:
        image_path = os.path.join(image_dir_path, image_file)
        frame = cv2.imread(image_path)
        out.write(frame)
    
    # Release VideoWriter and close any OpenCV windows
    out.release()
    cv2.destroyAllWindows()

def jrdb_helper(seq_len=250):
    # image_dir_path = "/pasteur/u/bencliu/open_surg/data/jrdb/images/image_0/clark-center-intersection-2019-02-28_0"
    video_save_path = "/pasteur/u/bencliu/open_surg/pips2/logs_demo/long_videos/clark_inters.mp4"
    # create_mp4_from_images(image_dir_path, video_save_path)

    logs_dir = "/pasteur/u/bencliu/open_surg/pips2/logs_demo/long_videos"
    main(filename=video_save_path, #Try lowering N and iters 
            S=seq_len, # seqlen -- OOM Triggers (500, 200, 150) || avoid triggers (58, )
            N=1024, # number of points per clip - prevously 1024
            stride=8, # spatial stride of the model
            timestride=1, # temporal stride of the model
            iters=4, # inference steps of the model
            log_dir=logs_dir, 
        ) 


def generate_point_clusters(points, num_points=40, max_radius=6):
    # Initialize an empty list to store the resulting points
    clustered_points = []

    # Loop through each point in the input list
    for x, y in points:
        # Generate 'num_points' points around the current point
        for _ in range(num_points):
            # Generate random angle and distance within the specified radius
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(0, max_radius)

            # Calculate the new point's coordinates
            new_x = x + distance * math.cos(angle)
            new_y = y + distance * math.sin(angle)

            # Add the new point to the clustered points list
            clustered_points.append([new_x, new_y])

    return clustered_points

def memory_screening():
    subset = range(300, 2000, 100) 
    for seq_len in subset:
        print("trying sequence length: ", seq_len)
        try:
            jrdb_helper(seq_len)
        except:
            print("failed on seq len: ", seq_len) 
            return 

if __name__ == '__main__':
    print("Starting demo script") 
    main() 
    debug()
    #surgical_analysis(log_dir="/pasteur/u/bencliu/open_surg/results/full_surg_set_seq200")
    #jrdb_helper(seq_len=200)
    #surgical_analysis("/pasteur/u/bencliu/open_surg/results/keypoint_clusters", True) 
    #surgical_analysis("/pasteur/u/bencliu/open_surg/results/full_surg_set") 
    #Fire(main)


""" Archived Code  


def frames_to_mp4(frame_dir):
    pass 

def initialize_point_grid(video_metadata):
    pass 

def run_model(model, rgbs, S_max=128, N=64, iters=16, sw=None, custom_grid=False, video_metadata=None):
    rgbs = rgbs.cuda().float() # B, S, C, H, W

    B, S, C, H, W = rgbs.shape
    assert(B==1) #B=1 during inference 

    # pick N points to track; we'll use a uniform grid | TODO - Initialize point cluster 
    N_ = np.sqrt(N).round().astype(np.int32)
    grid_y, grid_x = utils.basic.meshgrid2d(B, N_, N_, stack=False, norm=False, device='cuda')
    grid_y = 8 + grid_y.reshape(B, -1)/float(N_-1) * (H-16)
    grid_x = 8 + grid_x.reshape(B, -1)/float(N_-1) * (W-16)

    # Initialize uniform or customized grid 
    if custom_grid:
        xy0 = initialize_point_grid(video_metadata)
    else:
        xy0 = torch.stack([grid_x, grid_y], dim=-1) # B, N_*N_, 2  TODO - Initialize grid here
    _, S, C, H, W = rgbs.shape

    # zero-vel init
    trajs_e = xy0.unsqueeze(1).repeat(1,S,1,1) #TODO - Uncover with the S parameter is in this context 

    # print("Inspect input shapes") 
    # breakpoint() 

    iter_start_time = time.time()
    
    preds, preds_anim, _, _ = model(trajs_e, rgbs, iters=iters, feat_init=None, delta_mult=0.5)
    trajs_e = preds[-1]

    # print("Inspect output preds and trajectories") 
    # breakpoint() 

    iter_time = time.time()-iter_start_time
    print('inference time: %.2f seconds (%.1f fps)' % (iter_time, S/iter_time))

    if sw is not None and sw.save_this:
        rgbs_prep = utils.improc.preprocess_color(rgbs)
        sw.summ_traj2ds_on_rgbs('outputs/trajs_on_rgbs', trajs_e[0:1], utils.improc.preprocess_color(rgbs[0:1]), cmap='hot', linewidth=1, show_dots=False)
    return trajs_e

 
# S - Number of freames in sequence 
# N - Number of points to track usins uniform square grid over full iamge 
# [?] stride - Spatial stride
# [?] timestride - Temporal stride 
# [?] iters - Involved in forward pass => 


def main(
        filename='./stock_videos/camel.mp4',
        S=48, # seqlen
        N=1024, # number of points per clip
        stride=8, # spatial stride of the model
        timestride=1, # temporal stride of the model
        iters=16, # inference steps of the model
        image_size=(512,896), # input resolution
        max_iters=40, # number of clips to run, defaulted to 4 
        shuffle=False, # dataset shuffling
        log_freq=1, # how often to make image summaries
        log_dir='./logs_demo',
        init_dir='./reference_model',
        device_ids=[0],
        custom_points=None, 
):

    # the idea in this file is to run the model on a demo video,
    # and return some visualizations
    
    exp_name = 'de00' # copy from dev repo

    print('filename', filename)
    name = Path(filename).stem
    print('name', name)
    
    rgbs = read_mp4(filename)
    rgbs = np.stack(rgbs, axis=0) # S,H,W,3
    rgbs = rgbs[:,:,:,::-1].copy() # BGR->RGB
    rgbs = rgbs[::timestride]
    S_here,H,W,C = rgbs.shape
    print('rgbs', rgbs.shape)

    # autogen a name
    model_name = "%s_%d_%d_%s" % (name, S, N, exp_name)
    import datetime
    model_date = datetime.datetime.now().strftime('%H:%M:%S')
    model_name = model_name + '_' + model_date
    print('model_name', model_name)

    writer_t = SummaryWriter(log_dir + '/' + model_name + '/t', max_queue=10, flush_secs=60)

    global_step = 0

    # initialize model for inference 
    model = Pips(stride=8).cuda()
    parameters = list(model.parameters())
    if init_dir:
        _ = saverloader.load(init_dir, model)
    global_step = 0
    model.eval()

    idx = list(range(0, max(S_here-S,1), S))
    if max_iters:
        idx = idx[:max_iters]
    
    for si in idx: #Looping through clips => si provides starting index for clip 
        global_step += 1
        
        iter_start_time = time.time()

        #Logic for how the video is ultimately saved 
        sw_t = utils.improc.Summ_writer( 
            writer=writer_t,
            global_step=global_step,
            log_freq=log_freq,
            fps=8,
            scalar_freq=int(log_freq/2),
            just_gif=True)

        rgb_seq = rgbs[si:si+S] 
        rgb_seq = torch.from_numpy(rgb_seq).permute(0,3,1,2).to(torch.float32) # S,3,H,W
        rgb_seq = F.interpolate(rgb_seq, image_size, mode='bilinear').unsqueeze(0) # 1,S,3,H,W => Downsample to needed image size

        with torch.no_grad():
            trajs_e = run_model(model, rgb_seq, S_max=S, N=N, iters=iters, sw=sw_t)

        iter_time = time.time()-iter_start_time
        
        print('%s; step %06d/%d; itime %.2f' % (
            model_name, global_step, len(idx), iter_time))
        
    writer_t.close()




"""