import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import threading
import multiprocessing
import math

root_flow_path = './dataset/ucf-flow'

def cal_for_frames(video_path):
    capture = cv2.VideoCapture(video_path)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    _, prev = capture.read()
    prev = cv2.UMat(prev)
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    flow = []
    for count in tqdm(range(1, frame_count)):
        retaining, frame = capture.read()
        frame = cv2.UMat(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr = frame
        tmp_flow = compute_TVL1(prev, curr)
        flow.append(tmp_flow)
        prev = curr

    return flow


def compute_TVL1(prev, curr, bound=15):
    """Compute the TV-L1 optical flow."""
    TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
    flow = TVL1.calc(prev, curr, None)

    flow = cv2.UMat.get(flow)
    assert flow.dtype == np.float32

    flow = (flow + bound) * (255.0 / (2 * bound))
    flow = np.round(flow).astype(int)
    flow[flow >= 255] = 255
    flow[flow <= 0] = 0

    return flow


def save_flow(video_flows, flow_path):
    for i, flow in enumerate(video_flows):
        if not os.path.exists(flow_path+'_u'):
            print(flow_path+'_u')
        cv2.imwrite(os.path.join(flow_path+'_u', "{:06d}.jpg".format(i)),
                    flow[:, :, 0])
        cv2.imwrite(os.path.join(flow_path+'_v', "{:06d}.jpg".format(i)),
                    flow[:, :, 1])


def extract_flow(video_path, flow_path):
    flow = cal_for_frames(video_path)
    save_flow(flow, flow_path)
    return


def sort(video_paths):
    for x in tqdm(range(len(video_paths))):
        tar_path_u = os.path.join(root_flow_path, video_paths[x].split('/')[-3], video_paths[x].split('/')[-2],
                                  os.path.splitext(video_paths[x].split('/')[-1])[0] + '_u')
        tar_path_v = os.path.join(root_flow_path, video_paths[x].split('/')[-3], video_paths[x].split('/')[-2],
                                  os.path.splitext(video_paths[x].split('/')[-1])[0] + '_v')
        tar_path = os.path.join(root_flow_path, video_paths[x].split('/')[-3], video_paths[x].split('/')[-2],
                                os.path.splitext(video_paths[x].split('/')[-1])[0])
        if not os.path.exists(tar_path_u):
            os.makedirs(tar_path_u)
        if not os.path.exists(tar_path_v):
            os.makedirs(tar_path_v)
        flow_paths = tar_path
        extract_flow(video_paths[x], flow_paths)
    return

if __name__ == '__main__':
    thread_num = 16
    root_video_path = './dataset/ucf101'
    video_paths = glob(root_video_path+'/*/*/*.avi')
    print(len(video_paths))
    n = int(math.ceil(len(video_paths) / float(thread_num)))
    pool = multiprocessing.Pool(processes=thread_num)
    result = []
    for i in tqdm(range(0, len(video_paths), n)):
        result.append(pool.apply_async(sort, (video_paths[i: i+n],)))
    pool.close()
    pool.join()
    print('finish!')

