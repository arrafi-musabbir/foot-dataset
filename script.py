import pyvista as pv
import streamlit as st
from stpyvista import stpyvista
import os 
import glob
import streamlit as st
import os
# import helper
import pymeshlab
import shutil
from statistics import mean
import build_model
import keras
from keras import layers
from keras import ops
import numpy as np
import glob
import trimesh
from tensorflow import data as tf_data


os.makedirs('outputs', exist_ok=True)
os.makedirs('uploaded_file', exist_ok=True)
os.makedirs('temp_files', exist_ok=True)
os.makedirs('pre-filtered', exist_ok=True)


st.sidebar.title("MESH PROCESSING API")


@st.cache_resource 
def loading_all_models():
    model = build_model.build_model()
    return model

model = loading_all_models()


def saveUpload(fpath, tfile):
    with open(fpath,"wb") as f:
        f.write(tfile.getbuffer())
    return True

# if helper.api_check() == 200:
#     st.sidebar.write("API STATUS CHECK: 🟢")
# else:
#     st.sidebar.write("API STATUS CHECK: 🔴")

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

def manual_filter(file_path):
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
        if info_dict[key][0] > mean(nv) and info_dict[key][1] > mean(nf):
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(key)
            component_mesh_path = f"pre-filtered/{key.split('/')[-1]}"
            ms.save_current_mesh(component_mesh_path)
            component_mesh_path = f"pre-filtered/{key.split('/')[-1][:-4]}.obj"
            ms.save_current_mesh(component_mesh_path)
            component_mesh_path = f"pre-filtered/{key.split('/')[-1][:-4]}.stl"
            ms.save_current_mesh(component_mesh_path)
    return 'temp_files'

def show_objs_in_directory(fpath, directory_path):
    st.title("OBJ File Downloader")
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
        obj_files = sorted(glob.glob(os.path.join(directory_path, '*.obj')))
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

show_objs_in_directory('uploaded_file/uploaded_mesh.obj', 'temp_files')