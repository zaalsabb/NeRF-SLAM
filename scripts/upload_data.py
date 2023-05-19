import cv2
import requests
import numpy as np
import yaml
import json
import os
import sys

def send_query_image(url, I, D, pose, image_id, project_id=1):

    data1 = cv2.imencode('.jpg', I)[1].tobytes()
    data2 = cv2.imencode('.png', D)[1].tobytes()

    files = {'image':data1, 'depth':data2, 'pose':pose.tobytes()}
    endpoint = url + f'/api/v1/project/{project_id}/send_ref_image/{image_id}'
    response = requests.post(endpoint, files=files) 
    # response_json = response.json() 
    print(response)

def send_query_intrinsics(url, intrinsics, project_id=1):
    if url is None:
        return        
    headers = {
        'Content-Type': 'application/json',
    }        
    files = intrinsics

    endpoint = url + f'/api/v1/project/{project_id}/send_intrinsics'
    response = requests.post(endpoint, json=files, headers=headers) 
    print(response)

def load_project(url, project_id=1):
    if url is None:
        return False
    
    endpoint = url + f'/api/v1/project/{project_id}/load'
    response = requests.get(endpoint) 
    print(response)
    return True

def run_nerf(url, project_id=1):
    if url is None:
        return False
    
    endpoint = url + f'/api/v1/project/{project_id}/run_nerf'
    response = requests.get(endpoint) 
    print(response)
    return True


def load_args():

    with open('config/params.yaml') as f:
        args = yaml.safe_load(f)
    return args

def main(dataset_dir, url):
    
    url = 'http://'+ url+':5000/nerfslam'

    load_project(url)

    f_intrinsics = dataset_dir + '/intrinsics.json'
    with open(f_intrinsics) as f:
        intrinsics = json.load(f)

    poses = np.loadtxt(dataset_dir + '/poses.csv', delimiter=',')

    send_query_intrinsics(url, intrinsics)

    N = len(os.listdir(os.path.join(dataset_dir , 'rgb')))
    # N = 500

    for i in range(1,N):
        f_image = os.path.join(dataset_dir , 'rgb', f'{i:06}.jpg')
        # f_image = os.path.join(dataset_dir , 'rgb', f'{i}.png')
        I = cv2.imread(f_image)

        f_depth = os.path.join(dataset_dir , 'depth', f'{i:06}.png')
        # f_image = os.path.join(dataset_dir , 'rgb', f'{i}.png')
        D = cv2.imread(f_depth, cv2.IMREAD_UNCHANGED)

        pose = poses[i-1, 1:]
        send_query_image(url, I, D, pose, i-1)

    run_nerf(url)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])