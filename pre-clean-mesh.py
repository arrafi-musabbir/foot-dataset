import glob
import os
import pymeshlab
from statistics import mean

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


data_files = glob.glob("OBJ_A/*.obj")
print(data_files)
# data_files = ['OBJ_A/Foot_Left.obj']

for file in data_files[:]:
    clean_directory('temp_files')
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
        # res = ms.apply_filter('get_geometric_measures')
        
    # print(mean(nv), mean(nf))
    for key in info_dict:
        if info_dict[key][0] > mean(nv) and info_dict[key][1] > mean(nf):
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(key)
            component_mesh_path = f"{new_dir}/{key.split('/')[-1]}"
            ms.save_current_mesh(component_mesh_path)
        