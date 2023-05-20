import requests
import numpy as np
import yaml
from scipy.spatial.transform import Rotation


def send_pose(url, intrinsics, project_id=1):
    if url is None:
        return        
    headers = {
        'Content-Type': 'application/json',
    }        
    files = intrinsics

    endpoint = url + f'/api/v1/project/{project_id}/create_nerf_view'
    response = requests.post(endpoint, json=files, headers=headers) 
    print(response)


def load_args():

    with open('config/params.yaml') as f:
        args = yaml.safe_load(f)
    return args

def main():
    args = load_args()
    dataset_dir = args['dataset_dir']
    url = 'http://'+ args['url']+'/nerfslam'

    # pose = {'pose':[0.00995646, -0.47473243, -0.47473243,  0.94340211,  0.07508623, -0.04807876,  0.31944163]}
    pose = {'pose':[0,0,0, 0,0,0,1]}
    theta = 0
    d_theta = -10
    while True:
        q = Rotation.from_euler('zyx',[theta*np.pi/180,0,0]).as_quat()
        pose['pose'][3:] = q
        send_pose(url, pose)
        theta += d_theta

if __name__ == '__main__':
    main()