import cv2
import requests
import numpy as np
import yaml
import json
import os

def send_query_image(url, I, image_id, project_id=1):

    data = cv2.imencode('.jpg', I)[1].tobytes()
    files = {'image':data}
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

def load_args():

    with open('config/params.yaml') as f:
        args = yaml.safe_load(f)
    return args

def main():
    args = load_args()
    dataset_dir = args['dataset_dir']
    url = 'http://'+ args['url']+'/nerfslam'

    load_project(url)

    f_intrinsics = dataset_dir + '/intrinsics.json'
    with open(f_intrinsics) as f:
        intrinsics = json.load(f)

    send_query_intrinsics(url, intrinsics)

    N = len(os.listdir(os.path.join(dataset_dir , 'rgb')))

    for i in range(1,N):
        f_image = os.path.join(dataset_dir , 'rgb', f'{i}.png')
        I = cv2.imread(f_image)
        send_query_image(url, I, i)

if __name__ == '__main__':
    main()