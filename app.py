import os
import flask
from threading import Thread
import sys
from nerf_slam import NerfSLAM
import numpy as np
import PIL.Image as Image
import io
from icecream import ic


app = flask.Flask(__name__)
nerfs = {}
    
def load_nerf(project_id: int):     
    dataset_dir = os.path.join(f"/datasets", f"project_{project_id}")
    os.makedirs(dataset_dir, exist_ok=True)
    
    nerf_slam = NerfSLAM(dataset_dir)

    return nerf_slam

app.logger.info("NeRF SLAM server ready")

@app.route("/nerfslam/api/v1/project/<int:project_id>/load")
def load_project(project_id):
    nerf = load_nerf(project_id)
    if nerf is None:
        return flask.make_response("Project not found", 404)
    nerfs[project_id] = nerf
    print(f'Loaded project {project_id}')    
    return flask.make_response("Project loaded successfully")

@app.route("/nerfslam/api/v1/project/<int:project_id>/send_intrinsics", methods=["POST"])
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

@app.route("/nerfslam/api/v1/project/<int:project_id>/send_ref_poses", methods=["POST"])
def send_ref_poses(project_id):
    # recieve poses to scale of reference images
    # use these poses to correct the scale of the nerf model and localize it in the scene 
    pass

@app.route("/nerfslam/api/v1/project/<int:project_id>/send_ref_image/<int:image_id>", methods=["POST"])
def send_ref_image(project_id, image_id):
    # recieve image and save it to the project folder
    # keep track of image id
    if project_id not in nerfs:
        return flask.make_response("Project not loaded", 404)
    
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            img = Image.open(io.BytesIO(flask.request.files["image"].read()))    
            img = np.array(img)

            if flask.request.files.get("pose"):
                json_str=flask.request.files["pose"].read()
                pose = np.frombuffer(json_str, dtype="float")
                # ic(pose)
            else:
                return flask.make_response("Pose not found", 404)  

            nerfs[project_id].save_image(img, pose, image_id)
            return flask.make_response(f"Image {image_id} saved successfully")  
        else:
            return flask.make_response("Image not found", 404)            
    else:
        return flask.make_response("Invalid request", 404)

@app.route("/nerfslam/api/v1/project/<int:project_id>/run_nerf")
def run_nerf(project_id):
    if project_id not in nerfs:
        return flask.make_response("Project not loaded", 404)
    else:

        thread = Thread(
            target=run_nerf_background, args=(project_id,)
        )
        thread.start()

        return flask.make_response("Running nerf...")
    pass

@app.route("/nerfslam/api/v1/project/<int:project_id>/create_nerf_view", methods=["POST"])
def create_nerf_view(project_id):
    # receive pose
    # create nerf view
    # return rgb and depth images

    if flask.request.method == "POST":
        if "pose" in flask.request.json:
            pose = np.array(flask.request.json["pose"])  
            nerfs[project_id].create_view(pose)
            return flask.make_response("creating view...")
        else:
            return flask.make_response("pose not found", 404)
    else:
        return flask.make_response("Invalid request", 404)



def run_nerf_background(project_id):
    nerfs[project_id].run_nerf()


if __name__ == "__main__":
    #app.run(host='0.0.0.0',port=5000)
    app.run(host='::',port=5000, debug=True)    
