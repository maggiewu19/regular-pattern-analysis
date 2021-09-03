import cv2
import os 
import numpy as np 
import msgpack 
import argparse 
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

parser = argparse.ArgumentParser()
parser.add_argument('--fname', type=str, required=True)
args = parser.parse_args() 

def clip_video(video_dir, fname, start, end): 
    name = fname[-9:-4]
    targetname = video_dir + name + '_{}_{}.mp4'.format(start, end)
    ffmpeg_extract_subclip(fname, start, end, targetname=targetname)
    return targetname

def generate_frames(video_dir, frame_dir, fname): 
    name = fname[len(video_dir):-4]
    if not os.path.isdir(frame_dir + name):
        os.mkdir(frame_dir + name)

    vidcap = cv2.VideoCapture(fname)
    success, image = vidcap.read()
    count = 0

    while success:
        h, w = image.shape[:2]
        cv2.imwrite(frame_dir + name + "/frame%d.png" % count, image)    
        success,image = vidcap.read()
        if count % 100 == 0: 
            print('Read a new frame {}: '.format(count), success)
        count += 1

# def unpack(msg_file):
#     with open(msg_file, 'rb') as data_file:
#         data = msgpack.unpack(data_file, raw=False)
#     return data

# msg = video_dir + 'world.intrinsics'
# camera = unpack(msg)
# camera_matrix = np.array(camera['(1280, 720)']['camera_matrix'])
# resolution = camera['(1280, 720)']['resolution']
# dist_coefs = np.array(camera['(1280, 720)']['dist_coefs'])

start = 0
end = 5*60+14

fname = args.fname 
targetname = clip_video(video_dir, fname, start, end)
generate_frames(video_dir, frame_dir, targetname)
