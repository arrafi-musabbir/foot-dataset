import streamlit as st
import os
import helper
import pymeshlab
import shutil
from statistics import mean
import build_model
import keras
from keras import layers
import numpy as np
import glob
# import trimesh
from tensorflow import data as tf_data
from stpyvista import stpyvista
import pyvista as pv
import subprocess
import urllib.parse as parse

os.makedirs('outputs', exist_ok=True)
os.makedirs('uploaded_file', exist_ok=True)
os.makedirs('temp_files', exist_ok=True)
os.makedirs('pre-filtered', exist_ok=True)


def is_embed():
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    ctx = get_script_run_ctx()
    query_params = parse.parse_qs(ctx.query_string)
    return True if query_params.get("embed") else False

IS_APP_EMBED = is_embed()

st.sidebar.title("MESH PROCESSING API")



## Check if xvfb is already running on the machine
is_xvfb_running = subprocess.run(["pgrep", "Xvfb"], capture_output=True)

if is_xvfb_running.returncode == 1:
    if not IS_APP_EMBED:
        st.toast("Xvfb was not running...", icon="⚠️")
    pv.start_xvfb()
else:
    if not IS_APP_EMBED:
        st.toast(f"Xvfb is running! \n\n`PID: {is_xvfb_running.stdout.decode('utf-8')}`", icon="📺")

def clean_directory(directory_path):
    if os.path.exists(directory_path):
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

import tensorflow as tf



@st.cache_resource 
def loading_all_models():
    clean_directory('temp_files')
    clean_directory('pre-filtered')
    
    @keras.saving.register_keras_serializable('OrthogonalRegularizer')
    class OrthogonalRegularizer(keras.regularizers.Regularizer):

        def __init__(self, num_features, **kwargs):
            self.num_features = num_features
            self.l2reg = 0.001

        def call(self, x):
            x = tf.reshape(x, (-1, self.num_features, self.num_features))
            xxt = tf.tensordot(x, x, axes=(2, 2))
            xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
            eye = tf.eye(self.num_features)
            return tf.math.reduce_sum(self.l2reg * tf.square(xxt - eye))


        def get_config(self):
            return {"num_features": self.num_features, "l2reg": self.l2reg}
        
    model = keras.models.load_model('my_model.keras', custom_objects={'OrthogonalRegularizer': OrthogonalRegularizer})
    return model

model = loading_all_models()




if helper.api_check() == 200:
    st.sidebar.write("API STATUS CHECK: 🟢")
else:
    st.sidebar.write("API STATUS CHECK: 🔴")

def auto_filter(fpath):
    NUM_POINTS = 2048
    NUM_CLASSES = 1
    BATCH_SIZE = 64
    test_points1, test_labels1, CLASS_MAP1 = build_model.parse_unknown_dataset(fpath, NUM_POINTS)

    test_dataset1 = tf_data.Dataset.from_tensor_slices((test_points1, test_labels1))
    test_dataset1 = test_dataset1.shuffle(len(test_points1)).batch(BATCH_SIZE)

    data = test_dataset1.take(1)

    points, labels = list(data)[0]

    preds = model.predict(points)
    f = preds
    preds = tf.math.argmax(preds, -1)
    points = points.numpy()
    
    for i in range(test_points1.shape[0]):
        if f[i]==f.max():
            print("\tpred: {:}, \tlabel: {}, \tMAX FOOT".format(f[i], CLASS_MAP1[labels.numpy()[i]]))
            os.rename(f'pre-filtered/{CLASS_MAP1[labels.numpy()[i]][:-4]}.obj', f'pre-filtered/1foot_{CLASS_MAP1[labels.numpy()[i]][:-4]}.obj')
            # if CLASS_MAP1[labels.numpy()[i]].split(".")[0][-4:] == 'foot':
            #     global count
            #     count = count + 1
            #     print()
            # else:
            #     print("__________incorrect__________)")
        elif f[i]==f.min():
            print("\tpred: {:}, \tlabel: {}, \tMIN".format(f[i], CLASS_MAP1[labels.numpy()[i]]))
            os.rename(f'pre-filtered/{CLASS_MAP1[labels.numpy()[i]][:-4]}.obj', f'pre-filtered/2foot_{CLASS_MAP1[labels.numpy()[i]][:-4]}.obj')
        else:
            print("\tpred: {:}, \tlabel: {}".format(f[i], CLASS_MAP1[labels.numpy()[i]]))


def manual_filter(file_path):
    from datetime import datetime

    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    print(f"Current Time = {current_time}")
    
    file = file_path
    clean_directory('temp_files')
    clean_directory('pre-filtered')
    info_dict = {}
    nv = []
    nf = []
    new_dir = 'test-files/'+file.split('/')[-1][:-4]
    os.makedirs(new_dir, exist_ok=True)
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(file)
    ms.apply_filter('generate_splitting_by_connected_components')
    # print("\ttotal number of splited components: ", ms.mesh_number())
    for i in range(1, ms.mesh_number()):
        # print("\nObject: ",i)
        ms.set_current_mesh(i)
        # Step 2: Get mesh information
        num_vertices = ms.current_mesh().vertex_number()
        num_faces = ms.current_mesh().face_number()
        nv.append(num_vertices)
        nf.append(num_faces)
        # Step 3: Print mesh information
        # print("\tNumber of vertices:", num_vertices)
        # print("\tNumber of faces:", num_faces)
        component_mesh_path = f"temp_files/component_{i}.off"
        info_dict[component_mesh_path] = [num_vertices, num_faces]
        # print(component_mesh_path)
        # selected_mesh[component_mesh_path] = [num_vertices, num_faces]
        ms.save_current_mesh(component_mesh_path)
        component_mesh_path = f"temp_files/component_{i}.obj"
        ms.save_current_mesh(component_mesh_path)
        component_mesh_path = f"temp_files/component_{i}.stl"
        ms.save_current_mesh(component_mesh_path)
        # res = ms.apply_filter('get_geometric_measures')
    for key in info_dict:
        if info_dict[key][0] >= mean(nv) and info_dict[key][1] >= mean(nf):
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(key)
            component_mesh_path = f"pre-filtered/{key.split('/')[-1]}"
            ms.save_current_mesh(component_mesh_path)
            component_mesh_path = f"pre-filtered/{key.split('/')[-1][:-4]}.obj"
            ms.save_current_mesh(component_mesh_path)
            component_mesh_path = f"pre-filtered/{key.split('/')[-1][:-4]}.stl"
            ms.save_current_mesh(component_mesh_path)
    auto_filter('pre-filtered')
        
def order_mesh(directory_path):
    obj_files = glob.glob(os.path.join(directory_path, '*foot_*.obj'))
    info_dict = {}
    for file in obj_files:
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(file)
        # Step 2: Get mesh information
        num_vertices = ms.current_mesh().vertex_number()
        num_faces = ms.current_mesh().face_number()
        # Step 3: Print mesh information
        # print("\tNumber of vertices:", num_vertices)
        # print("\tNumber of faces:", num_faces)
        info_dict[file] = [num_vertices, num_faces]
    sorted_data = dict(sorted(info_dict.items(), key=lambda item: item[1][1], reverse=True))
    print(sorted_data)
    return sorted_data.keys()


def show_objs_in_directory(fpath, directory_path):
    st.title("OBJ File Downloader")
    obj_files = sorted(glob.glob(os.path.join('temp_files', '*.obj')))
    selected_obj = st.sidebar.selectbox(f"Total Seperated Components: {len(obj_files)}\n\n from manual-filter", obj_files)
    col1, col2 = st.columns(2)
    
    with col1:
        
        f = [fpath.split('/')[-1]]
        selected_obj = st.selectbox("Select an OBJ file to view:", f)
        try:
            plotter = pv.Plotter(window_size=[600, 400])
            mesh = pv.read(fpath)
            plotter.add_mesh(mesh, cmap='gray', line_width=1)
            plotter.view_isometric()
            plotter.background_color = 'white'
            stpyvista(plotter, use_container_width=True)
        except Exception as e:
            print(f"Error reading or displaying the selected OBJ file: {e}")
            st.error(f"Error reading or displaying the selected OBJ file: {e}")
            
    with col2:
        # obj_files = sorted(glob.glob(os.path.join(directory_path, '*foot_*.obj')), reverse=True)
        obj_files = order_mesh(directory_path)
        if not obj_files:
            st.warning("No OBJ files found in the directory.")
            return
        selected_obj = st.selectbox("Select an OBJ file to view:", obj_files)
        
        try:
            plotter = pv.Plotter(window_size=[600, 400])
            mesh = pv.read(selected_obj)
            plotter.add_mesh(mesh, cmap='gray', line_width=1)
            plotter.view_isometric()
            plotter.background_color = 'white'
            stpyvista(plotter, use_container_width=True)
        except Exception as e:
            print(f"Error reading or displaying the selected OBJ file: {e}")
            st.error(f"Error reading or displaying the selected OBJ file: {e}")

        st.write("")  # Add some space
        obj_file_bytes = open(selected_obj, 'rb').read()
        st.download_button(
            label="Download Selected OBJ",
            data=obj_file_bytes,
            file_name=os.path.basename(selected_obj),
            key="download_button_1",
            mime="text/plain",  # Specify the MIME type as text/plain for OBJ files
        )
        # zip_files = sorted(glob.glob(os.path.join(directory_path, '*.zip')))
        # # Add a download button for the selected OBJ file
        # st.write("")  # Add some space
        # obj_file_bytes = open(zip_files[0], 'rb').read()
        # st.download_button(
        #     label="Download ALL ZIP",
        #     data=obj_file_bytes,
        #     file_name=os.path.basename(zip_files[0]),
        #     key="download_button_2",
        #     mime="text/plain",  # Specify the MIME type as text/plain for OBJ files
        # )
        
def saveUpload(fpath, tfile):
    clean_directory("uploaded_file")
    clean_directory("pre-filtered")
    clean_directory("temp_files")
    with open(fpath,"wb") as f:
        f.write(tfile.getbuffer())
    return True

def reconstruct_mesh(fpath):
    # os.makedirs('reconstructed', exist_ok=True)
    clean_directory("pre-filtered")
    # fpath = os.path.join("uploaded_file", 'uploaded_mesh.obj')
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(fpath)
    ms.apply_filter('generate_surface_reconstruction_screened_poisson', visiblelayer=True, preclean=True, threads=1)
    output_mesh_path = os.path.join('pre-filtered', f"reconstructed_foot_temp.obj")
    ms.save_current_mesh(output_mesh_path)

def mirror_mesh(fpath):
    # os.makedirs('reconstructed', exist_ok=True)
    clean_directory("pre-filtered")
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(fpath)
        # Define axis mapping for mirroring
        # if axis =='x':
    flip_x, flip_y, flip_z = True, False, False
        # if axis =='y':
        #     flip_x, flip_y, flip_z = False, True, False
        # if axis =='z':
        #     flip_x, flip_y, flip_z = False, False, True
    axis = 'x'      
    # Apply the mirror transformation
    ms.apply_filter('apply_matrix_flip_or_swap_axis', flipx=flip_x, flipy=flip_y, flipz=flip_z)
    ms.apply_filter('meshing_invert_face_orientation', forceflip = True)
    output_mesh_path = os.path.join('pre-filtered', f"mirrored_foot_temp_{axis}_axis.obj")
    # Save the mirrored mesh
    ms.save_current_mesh(output_mesh_path)

def remesh(fpath):
    path = 'autoremesher.AppImage'
    print("remesh starting!")
    clean_directory("pre-filtered")
    input_path = fpath
    temp_path = 'pre-filtered' + '/remesh_foot_temp.obj'
    output_path = os.path.join(os.getcwd(), temp_path)
    print("input path: ", input_path)
    print("output path: ", output_path)
    print()

    p = subprocess.Popen([f'chmod a+x ./{path}'], shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) 
    out, err = p.communicate()
    print('SUBPROCESS ERROR: ' + str(err))
    print('SUBPROCESS stdout: ' + str(out.decode())) 
    print("")

    # command = f"./quadremesher/autoremesher.AppImage --help"
    # command = f"./{path} -i {input_path} -o {output_path}"
    # command = f"./{path}  --appimage-extract && ./squashfs-root/AppRun --help && rm -rf ./squashfs-root"
    command = f"./{path}  --appimage-extract && ./squashfs-root/AppRun -i {input_path} -o {output_path} && rm -rf ./squashfs-root"

    process = subprocess.Popen([command], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
            # st.write(output.strip())

uploaded_file = st.sidebar.file_uploader("Choose a mesh file (.obj)", type='obj')
if uploaded_file is not None:
    # helper.clean_directory('outputs')
    # helper.clean_directory('uploaded_file')
    fpath = os.path.join("uploaded_file", 'uploaded_mesh.obj')
    fpath = os.path.join(os.getcwd(), fpath)
    saveUpload(fpath, uploaded_file)
    print("\n")
    # manual_filter(fpath)
    

fpath = os.path.join("uploaded_file", 'uploaded_mesh.obj')
remesh(fpath)
# st.sidebar.button('ai-filter', on_click=manual_filter, args = [fpath])
# st.sidebar.button('mirror-mesh-object', on_click=mirror_mesh, args= [fpath])
# # st.sidebar.button('filter-mesh-artifacts', on_click=filter_artifacts, args= [fpath])
# st.sidebar.button('surface-reconstruction', on_click=reconstruct_mesh, args= [fpath])

# show_objs_in_directory(fpath, 'pre-filtered') 
    