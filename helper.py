import subprocess
import os
import requests
import streamlit as st
import pyvista as pv
from stpyvista import stpyvista
import glob
import shutil 

url0 = 'https://musabbir.pythonanywhere.com/'
url1 = 'https://musabbir.pythonanywhere.com/upload_mesh'
url2 = 'https://musabbir.pythonanywhere.com/filter_artifacts'
url3 = 'https://musabbir.pythonanywhere.com/mirror_mesh'
url4 = 'https://musabbir.pythonanywhere.com/remesh'
url5 = 'https://musabbir.pythonanywhere.com/reconstruct_mesh'

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

def show_objs_in_directory(fpath, directory_path):
    st.title("OBJ File Downloader")
    col1, col2 = st.columns(2)
    
    with col1:
        f = [fpath.split('/')[-1]]
        selected_obj = st.selectbox("Select an OBJ file to view:", f)
        # try:
        #     plotter = pv.Plotter(window_size=[600, 400])
        #     mesh = pv.read(fpath)
        #     plotter.add_mesh(mesh, cmap='gray', line_width=1)
        #     plotter.view_isometric()
        #     plotter.background_color = 'white'
        #     stpyvista(plotter, use_container_width=True)
        # except Exception as e:
        #     st.error(f"Error reading or displaying the selected OBJ file: {e}")
            
    with col2:
        obj_files = sorted(glob.glob(os.path.join(directory_path, '*.obj')))
        if not obj_files:
            st.warning("No OBJ files found in the directory.")
            return
        selected_obj = st.selectbox("Select an OBJ file to view:", obj_files)
        
        # try:
        #     plotter = pv.Plotter(window_size=[600, 400])
        #     mesh = pv.read(selected_obj)
        #     plotter.add_mesh(mesh, cmap='gray', line_width=1)
        #     plotter.view_isometric()
        #     plotter.background_color = 'white'
        #     stpyvista(plotter, use_container_width=True)
        # except Exception as e:
        #     st.error(f"Error reading or displaying the selected OBJ file: {e}")

        # Add a download button for the selected OBJ file
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
def api_check():
    try:
        response = requests.get(url0)
        return response.status_code
        # print(response, response.text)
    except requests.exceptions.RequestException as e:
        print(f'Error: {e}')

import select
def quad_remesh_conversion(file_path):
    clean_directory('outputs/')
    input_path = file_path
    temp_path =  'outputs/remesh.obj'
    output_path = os.path.join(os.getcwd(), temp_path)
    print("input path: ", input_path)
    print("output path: ", output_path)
    
    process = subprocess.Popen(['modprobe libfuse2'], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    import time
        # Display the output in Streamlit
    if stdout:
        st.text(f"Standard Output:\n{stdout}")
    if stderr:
        st.text(f"Standard Error:\n{stderr}")
    
    
    appimage_path = os.getcwd() + '/quadremesher/autoremesher.AppImage'
    cmd1 = f'chmod +x {appimage_path}'
    command = f"{appimage_path} -i {input_path} -o {output_path}"
    # st.write(command)
    subprocess.Popen([cmd1], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # process = subprocess.Popen([command], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # stdout, stderr = process.communicate()

    # import time
    # time.sleep(60)
    #     # Display the output in Streamlit
    # if stdout:
    #     st.text(f"Standard Output:\n{stdout}")
    # if stderr:
    #     st.text(f"Standard Error:\n{stderr}")

    # command = f"ls -la /mount/src/mesh-processing-api/"
    # process = subprocess.Popen([command], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # stdout, stderr = process.communicate()

        # Display the output in Streamlit
    # if stdout:
    #     st.text(f"Standard Output:\n{stdout}")
    # if stderr:
    #     st.text(f"Standard Error:\n{stderr}") 
        
    # command = f"ls -la /mount/src/mesh-processing-api/"
    # process = subprocess.Popen([command], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # stdout, stderr = process.communicate()

    # Ensure the process is finished and capture any remaining output
    # remaining_output, _ = process.communicate()
    # if remaining_output:
    #     st.write(remaining_output.strip())

import zipfile
def unzip_meshes(zip_file_path):
    # Specify the path to the ZIP file you want to unzip
    zip_file_path = zip_file_path  # Replace with the actual ZIP file path
    # Specify the directory where you want to extract the contents
    extracted_directory = 'outputs/'  # Replace with your desired extraction directory
    # Create the extraction directory if it doesn't exist
    import os
    os.makedirs(extracted_directory, exist_ok=True)

    # Open the ZIP file for reading
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # Extract all the contents of the ZIP file to the specified directory
        zip_ref.extractall(extracted_directory)

    print(f"Successfully extracted contents to {extracted_directory}")


def upload_mesh(file_path):
    clean_directory('outputs/')
    zip_file_path = 'outputs/uploaded_file.zip'
    try:
        with open(file_path, 'rb') as f:
            files = {'file': f}

            response = requests.post(url1, files=files, stream=True, timeout=120)
            if response.status_code == 200:
                # The endpoint should return a file. Write it to a local file.
                with open(zip_file_path, 'wb') as out_file:
                    for chunk in response.iter_content(chunk_size=4096):
                        out_file.write(chunk)
                print("Downloaded and saved uploaded file.")
            else:
                print(f"Filter-API Response: {response.status_code}, {response.text}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error: {e}")


def filter_artifacts(file_path):
    clean_directory('outputs/')
    zip_file_path = 'outputs/filtered_files.zip'
    try:
        with open(file_path, 'rb') as f:
            files = {'file': f}

            # Post to filter artifacts and save the returned file
            response = requests.post(url2, files=files, stream=True, timeout=120)
            if response.status_code == 200:
                # The endpoint should return a file. Write it to a local file.
                with open(zip_file_path, 'wb') as out_file:
                    for chunk in response.iter_content(chunk_size=4096):
                        out_file.write(chunk)
                print("Downloaded and saved filtered mesh file.")
            else:
                print(f"Filter-API Response: {response.status_code}, {response.text}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error: {e}")
    unzip_meshes(zip_file_path)


def mirror_mesh(file_path):
    clean_directory('outputs/')
    zip_file_path = 'outputs/mirrored_files.zip'
    try:
        with open(file_path, 'rb') as f:
            files = {'file': f}

            # Post to filter artifacts and save the returned file
            response = requests.post(url3, files=files, stream=True, timeout=360)
            if response.status_code == 200:
                # The endpoint should return a file. Write it to a local file.
                with open(zip_file_path, 'wb') as out_file:
                    for chunk in response.iter_content(chunk_size=4096):
                        out_file.write(chunk)
                print("Downloaded and saved mirrored mesh file.")
            else:
                print(f"Mirror-API Response: {response.status_code}, {response.text}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error: {e}")
    unzip_meshes(zip_file_path)

def reconstruct_mesh(file_path):
    clean_directory('outputs/')
    zip_file_path = 'outputs/reconstructed_mesh.zip'
    try:
        with open(file_path, 'rb') as f:
            files = {'file': f}
            # Post to filter artifacts and save the returned file
            response = requests.post(url5, files=files, stream=True, timeout=360)
            if response.status_code == 200:
                # The endpoint should return a file. Write it to a local file.
                with open(zip_file_path, 'wb') as out_file:
                    for chunk in response.iter_content(chunk_size=4096):
                        out_file.write(chunk)
                print("Downloaded and saved reconstructed mesh file.")
            else:
                print(f"RECONSTRUCTION-API Response: {response.status_code}, {response.text}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error: {e}")
    unzip_meshes(zip_file_path)