import os
import flask
from multiprocessing import Process
from threading import Thread
import sys
from nerf_slam import NerfSLAM
import numpy as np
import PIL.Image as Image
import io
from icecream import ic
import torch
import time
import json
import cv2
import zipfile

app = flask.Flask(__name__)
nerfs = {}
commands = {}
    
def load_nerf(project_id: int):     
    dataset_dir = os.path.join(f"/datasets", f"project_{project_id}")
    os.makedirs(dataset_dir, exist_ok=True)
    
    nerf_slam = NerfSLAM(dataset_dir)

    return nerf_slam

app.logger.info("NeRF SLAM server ready")

@app.route("/nerfslam/<int:project_id>/load")
def load_project(project_id):
    nerf = load_nerf(project_id)
    if nerf is None:
        return flask.make_response("Project not found", 404)
    nerfs[project_id] = nerf

    commands[project_id] = "load"

    print(f'Loaded project {project_id}')    
    return flask.make_response("Project loaded successfully")

@app.route("/nerfslam/<int:project_id>/send_intrinsics", methods=["POST"])
def send_intrinsics(project_id):
    # recieve camera matrix and save it to the project folder
    if flask.request.method == "POST":
        if "camera_matrix" in flask.request.json and \
            "width" in flask.request.json and \
            "height" in flask.request.json:
            K = np.array(flask.request.json["camera_matrix"])  
            d = np.array(flask.request.json["dist_coeff"])
            w = flask.request.json["width"]
            h = flask.request.json["height"]  
            nerfs[project_id].set_intrinsics(K, w, h, d)
            return flask.make_response("Camera intrinsics added successfully")  
        else:
            return flask.make_response("Camera intrinsics not found", 404)
    else:
        return flask.make_response("Invalid request", 404)


@app.route("/nerfslam/<int:project_id>/send_ref_image/<int:image_id>", methods=["POST"])
def send_ref_image(project_id, image_id):
    # recieve image and save it to the project folder
    # keep track of image id
    if project_id not in nerfs:
        return flask.make_response("Project not loaded", 404)
    
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            img = Image.open(io.BytesIO(flask.request.files["image"].read()))    
            img = np.array(img)

            # depth = Image.open(io.BytesIO(flask.request.files["depth"].read()))    
            # depth = np.array(depth)            

            if flask.request.files.get("pose"):
                json_str=flask.request.files["pose"].read()
                pose = np.frombuffer(json_str, dtype="float")
                # ic(pose)
            else:
                return flask.make_response("Pose not found", 404)  

            nerfs[project_id].save_image(img, None, pose, image_id)
            return flask.make_response(f"Image {image_id} saved successfully")  
        else:
            return flask.make_response("Image not found", 404)            
    else:
        return flask.make_response("Invalid request", 404)

@app.route("/nerfslam/<int:project_id>/run_nerf")
def run_nerf(project_id):
    # if project_id not in nerfs:
    #     return flask.make_response("Project not loaded", 404)

    # thread = Thread(target=run_nerf_background, args=(project_id,))
    # thread.start()
    # proc = Process(target=run_nerf_background, args=(project_id,))
    # proc.start()
    commands[project_id] = "run"

    return flask.make_response("Running nerf...")


@app.route("/nerfslam/<int:project_id>/create_nerf_view", methods=["POST"])
def create_nerf_view(project_id):
    # receive pose
    # create nerf view
    # return rgb and depth images

    if not flask.request.method == "POST":
        return flask.make_response("Invalid request", 404)
    
    if not "pose" in flask.request.json:
        return flask.make_response("pose not found", 404)

    dataset_dir = os.path.join(f"/datasets", f"project_{project_id}")
    output_dir = os.path.join(dataset_dir, 'output')

    pose_path = os.path.join(dataset_dir, 'query_pose.json')

    with open(pose_path, 'w') as f:
        json.dump(flask.request.json, f)      

    # watch for output file changes
    cached_stamp = None
    f_Twc = os.path.join(output_dir, 'T_w_c.txt')
    timeout = 0.5
    t1 = time.time()
    while time.time() - t1 < timeout:
        if not os.path.exists(f_Twc):
            continue
        
        if cached_stamp is None:
            cached_stamp = os.stat(f_Twc).st_mtime

        if os.stat(f_Twc).st_mtime != cached_stamp:
            break

    data = io.BytesIO()
    with zipfile.ZipFile(data, mode='w') as z:
        for f in os.listdir(output_dir):
            z.write(os.path.join(output_dir, f))

    data.seek(0)
    return flask.send_file(
        data,
        mimetype='application/zip',
        as_attachment=True,
        download_name='data.zip'
    )    
    

def run_nerf_background(project_id):
    nerf = load_nerf(project_id)
    nerfs[project_id] = nerf    

    nerfs[project_id].run_nerf()

def web():
    app.run(host='::',port=5002, debug=True, use_reloader=False) 

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)

    Thread(target=web, daemon=True).start()
    while True:

        for project_id in list(commands):

            com = commands[project_id]

            if com == "load":
                nerfs[project_id] = load_nerf(project_id)
            elif com == "run":
                # nerfs[project_id] = load_nerf(project_id)
                nerfs[project_id].run_nerf()                 

            commands.pop(project_id)
        time.sleep(1)    

    # app.run(host='::',port=5002, debug=True) 

