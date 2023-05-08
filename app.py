import os
import flask
from threading import Thread
import sys
from nerf_slam import NerfSLAM
import numpy as np
import PIL.Image as Image
import io

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
            w = flask.request.json["width"]
            h = flask.request.json["height"]  
            nerfs[project_id].set_intrinsics(K, w, h)
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
            nerfs[project_id].save_image(img, image_id)
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
        nerfs[project_id].run_nerf()
        return flask.make_response("Running nerf")
    pass

@app.route("/nerfslam/api/v1/project/<int:project_id>/create_nerf_view")
def create_nerf_view(project_id):
    # receive pose
    # create nerf view
    # return rgb and depth images
    pass


# @app.route("/api/v1/project/<int:project_id>/localize", methods=["POST"])
# def localize_request(project_id):
#     if project_id not in localizers:
#         return flask.make_response("Project not loaded", 404)
    
#     if flask.request.method == "POST":

#         if flask.request.files.get("camera_matrix"):
#             json_str=flask.request.files["camera_matrix"].read()
#             camera_matrix = np.frombuffer(json_str, dtype="float").reshape(3,3)
#         else:
#             return flask.make_response("Intrinsics not found", 404)

#         if flask.request.files.get("image"):
#             loc = localizers[project_id]

#             img = Image.open(io.BytesIO(flask.request.files["image"].read()))    
#             img = np.array(img)
            
#             T_m1_c2, inliers = loc.callback_query(img, camera_matrix)
#             if T_m1_c2 is None:
#                 res = {'success':False}
#             else:
#                 pose = matrix2pose(T_m1_c2)
#                 res = {'pose':tuple(pose.tolist()), 'inliers':inliers, 'success':True}

#             return flask.make_response(res)

#         else:
#             return flask.make_response("Image not found", 404)
#     else:
#         return flask.make_response("Invalid request", 404)

# @app.route("/api/v1/project/<int:project_id>/localize_multiple", methods=["POST"])
# def localize_multiple_request(project_id):
#     if project_id not in localizers:
#         return flask.make_response("Project not loaded", 404)
    
#     if flask.request.method == "POST":

#         N_imgs = len(flask.request.files) - 2
#         if N_imgs < 1:
#             return flask.make_response("Incorrect number of inputs", 404)
        
#         if flask.request.files.get("camera_matrix"):
#             json_str=flask.request.files["camera_matrix"].read()
#             camera_matrix = np.frombuffer(json_str, dtype="float").reshape(3,3)
#         else:
#             return flask.make_response("Intrinsics not found", 404)        

#         if flask.request.files.get("poses"):
#             json_str=flask.request.files["poses"].read()
#             poses_l = np.frombuffer(json_str, dtype="float").reshape(-1,7)
#             poses_l = poses_l.tolist()
#         else:
#             return flask.make_response("Poses not found", 404)  

#         loc = localizers[project_id]
#         I2_l = []
        
#         for fkey in flask.request.files.keys():
#             if "image" in fkey:            
#                 img = Image.open(io.BytesIO(flask.request.files[fkey].read()))    
#                 img = np.array(img)
#                 I2_l.append(img)

#         T_m1_m2, inliers = loc.callback_query_multiple(I2_l, poses_l, camera_matrix)
#         if T_m1_m2 is None:
#             res = {'success':False}
#         else:
#             pose = matrix2pose(T_m1_m2)
#             res = {'pose':tuple(pose.tolist()), 'inliers':inliers, 'success':True}

#         return flask.make_response(res)

#     else:
#         return flask.make_response("Invalid request", 404)

# # post request method for uploading data to local filesystem for development
# @app.route("/api/v1/project/<int:project_id>/upload", methods=["POST"])
# def upload(project_id):
#     if flask.request.method == "POST":
#         if len(flask.request.data) > 0:

#             with open(os.path.join("/Data.zip"), "wb") as f:
#                 f.write(flask.request.data)

#             home = os.environ.get('HOME')
#             sample_data = os.path.join(home, "datasets", f"project_{project_id}")
#             os.makedirs(sample_data, exist_ok=True)

#             thread = Thread(
#                 target=preprocess_task, args=(sample_data,project_id,)
#             )
#             thread.start()

#             return "success"
#         else:
#             return flask.make_response("Data not found", 404)
#     else:
#         return flask.make_response("Invalid request", 404)

# def preprocess_task(sample_data,project_id):
#     print("started preprocessing...")
#     shutil.rmtree(os.path.join("/Data"), ignore_errors=True)
#     with zipfile.ZipFile("/Data.zip", "r") as zip_ref:
#         zip_ref.extractall("/Data")
#     shutil.rmtree(sample_data, ignore_errors=True)

#     # remove output directory folders if they exist
#     frame_rate = 2
#     max_depth = 5
#     voxel = 0.01
#     # create preprocessor object
#     home = os.environ.get('HOME')
#     process = subprocess.Popen(["python3", "preprocessor/cli.py",
#                                 "-i", "/Data", "-o", sample_data, "-f", str(frame_rate), "-d", str(max_depth), "-v", str(voxel),
#                                 "--mobile_inspector"])
#     process.wait()

#     print('Reloading project...')  
#     loc_1 = load_localizer(project_id) 
#     loc_1.build_database()
#     localizers[project_id] = loc_1
#     intrinsics[project_id] = loc_1.camera_matrix     
#     print(f'Loaded project {project_id}')    

if __name__ == "__main__":
    #app.run(host='0.0.0.0',port=5000)
    app.run(host='::',port=5000, debug=True)    
