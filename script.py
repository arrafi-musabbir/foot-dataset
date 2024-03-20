import tensorflow as tf
import keras
import glob
import os 
import numpy as np
import pymeshlab
import trimesh
import shutil
from tensorflow import data as tf_data
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

def parse_unknown_dataset(file_path, num_points):
    test_points = []
    test_labels = []
    class_map = {}
    i = 0
    test_files = glob.glob(f"{file_path}/*.off")

    means = []
    stds = []

    for f in test_files:
        class_map[i] = f.split("/")[-1]
        point_cloud = trimesh.load(f).sample(num_points)

        mean = np.mean(point_cloud, axis=0)
        std = np.std(point_cloud, axis=0)

        normalized_point_cloud = (point_cloud - mean) / std

        test_points.append(normalized_point_cloud)
        test_labels.append(i)

        means.append(mean)
        stds.append(std)
        i += 1
        
    means = np.array(means)
    stds = np.array(stds)

    return (
        np.array(test_points),
        np.array(test_labels),
        class_map
    )
    
def loading_all_models():
  
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

def auto_filter(fpath):
    NUM_POINTS = 2048
    NUM_CLASSES = 1
    BATCH_SIZE = 32
    test_points1, test_labels1, CLASS_MAP1 = parse_unknown_dataset(fpath, NUM_POINTS)

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
    print(info_dict)
    for key in info_dict:
        if info_dict[key][0] > mean(nv) and info_dict[key][1] > mean(nf):
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(key)
            component_mesh_path = f"pre-filtered/{key.split('/')[-1]}"
            ms.save_current_mesh(component_mesh_path)
            component_mesh_path = f"pre-filtered/{key.split('/')[-1][:-4]}.obj"
            ms.save_current_mesh(component_mesh_path)
            # component_mesh_path = f"pre-filtered/{key.split('/')[-1][:-4]}.stl"
            # ms.save_current_mesh(component_mesh_path)
            
    auto_filter('pre-filtered')
    objs_files = order_mesh('pre-filtered')
    print(objs_files)
    
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

manual_filter('FullHead.obj')