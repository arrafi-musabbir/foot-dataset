import streamlit as st
import os
import helper
import pymeshlab
import shutil
from statistics import mean
import build_model
import keras
from keras import layers
from keras import ops
import numpy as np
import glob
# import trimesh
from tensorflow import data as tf_data
from stpyvista import stpyvista
import pyvista as pv


os.makedirs('outputs', exist_ok=True)
os.makedirs('uploaded_file', exist_ok=True)
os.makedirs('temp_files', exist_ok=True)
os.makedirs('pre-filtered', exist_ok=True)


st.sidebar.title("MESH PROCESSING API")


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


@st.cache_resource 
def loading_all_models():
    clean_directory('temp_files')
    clean_directory('pre-filtered')
    model = build_model.build_model()
    return model

model = loading_all_models()


def saveUpload(fpath, tfile):
    with open(fpath,"wb") as f:
        f.write(tfile.getbuffer())
    return True

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
    preds = ops.argmax(preds, -1)
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
    auto_filter('pre-filtered')
            


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


uploaded_file = st.sidebar.file_uploader("Choose a mesh file (.obj)", type='obj')
if uploaded_file is not None:
    # helper.clean_directory('outputs')
    # helper.clean_directory('uploaded_file')
    fpath = os.path.join("uploaded_file", 'uploaded_mesh.obj')
    saveUpload(fpath, uploaded_file)
    fpath =os.path.join(os.getcwd(), fpath)
 
    # st.sidebar.button('quad-conversion', on_click=helper.quad_remesh_conversion, args = [fpath])
    # st.sidebar.button('mirror-mesh-object', on_click=helper.mirror_mesh, args= [fpath])
    # st.sidebar.button('filter-mesh-artifacts', on_click=helper.filter_artifacts, args= [fpath])
    # st.sidebar.button('surface-reconstruction', on_click=helper.reconstruct_mesh, args= [fpath])
    a = st.sidebar.button('filter', on_click=manual_filter, args= [fpath])
    # st.sidebar.button('intelligent-filter', on_click=auto_filter, args= ['pre-filtered'])
    # st.sidebar.button('manual-filter', on_click=helper.reconstruct_mesh, args= [fpath])
    # obj_files = sorted(glob.glob(os.path.join('pre-filtered', '*.obj')))
    # if not obj_files:
    #     st.sidebar.warning("No OBJ files found in the directory.")
    # selected_obj = st.sidebar.selectbox("Select an OBJ file to view:", obj_files)
    # obj_file_bytes = open(selected_obj, 'rb').read()
    # st.sidebar.download_button(
    #     label="Download Selected OBJ",
    #     data=obj_file_bytes,
    #     file_name=os.path.basename(selected_obj),
    #     key="download_button_2",
    #     mime="text/plain",  )# Specify the MIME type as text/plain for OBJ files
    # if a:
    show_objs_in_directory(fpath, 'pre-filtered')
    # helper.show_objs_in_directory(fpath, 'outputs')
    