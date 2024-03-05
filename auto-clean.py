# import keras
# from keras import layers
# from keras import ops
# import numpy as np
# import glob
# import trimesh
# from tensorflow import data as tf_data


# print(keras.__version__)

# NUM_POINTS = 2048
# NUM_CLASSES = 1
# BATCH_SIZE = 64


# def conv_bn(x, filters):
#     x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
#     x = layers.BatchNormalization(momentum=0.0)(x)
#     return layers.Activation("relu")(x)


# def dense_bn(x, filters):
#     x = layers.Dense(filters)(x)
#     x = layers.BatchNormalization(momentum=0.0)(x)
#     return layers.Activation("relu")(x)

# class OrthogonalRegularizer(keras.regularizers.Regularizer):
#     def __init__(self, num_features, l2reg=0.001):
#         self.num_features = num_features
#         self.l2reg = l2reg
#         self.eye = ops.eye(num_features)

#     def __call__(self, x):
#         x = ops.reshape(x, (-1, self.num_features, self.num_features))
#         xxt = ops.tensordot(x, x, axes=(2, 2))
#         xxt = ops.reshape(xxt, (-1, self.num_features, self.num_features))
#         return ops.sum(self.l2reg * ops.square(xxt - self.eye))


# def tnet(inputs, num_features):
#     # Initalise bias as the indentity matrix
#     bias = keras.initializers.Constant(np.eye(num_features).flatten())
#     reg = OrthogonalRegularizer(num_features)

#     x = conv_bn(inputs, 32)
#     x = conv_bn(x, 64)
#     x = conv_bn(x, 512)
#     x = layers.GlobalMaxPooling1D()(x)
#     x = dense_bn(x, 256)
#     x = dense_bn(x, 128)
#     x = layers.Dense(
#         num_features * num_features,
#         kernel_initializer="zeros",
#         bias_initializer=bias,
#         activity_regularizer=reg,
#     )(x)
#     feat_T = layers.Reshape((num_features, num_features))(x)
#     # Apply affine transformation to input features
#     return layers.Dot(axes=(2, 1))([inputs, feat_T])

# inputs = keras.Input(shape=(NUM_POINTS, 3))

# x = tnet(inputs, 3)
# x = conv_bn(x, 32)
# x = layers.Dropout(0.5)(x)
# x = conv_bn(x, 32)
# x = tnet(x, 32)
# x = conv_bn(x, 32)
# x = conv_bn(x, 64)
# x = layers.Dropout(0.5)(x)
# x = conv_bn(x, 512)
# x = layers.GlobalMaxPooling1D()(x)
# x = dense_bn(x, 256)
# x = layers.Dropout(0.5)(x)
# x = dense_bn(x, 128)
# x = dense_bn(x, 128)
# x = dense_bn(x, 64)
# x = layers.Dropout(0.5)(x)
# x = dense_bn(x, 32)
# x = layers.Dropout(0.5)(x)

# outputs = layers.Dense(NUM_CLASSES, activation="sigmoid")(x)
# model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
# # model.summary()

# def parse_unknown_dataset(file_path, num_points):
#     test_points = []
#     test_labels = []
#     class_map = {}
#     i = 0
#     test_files = glob.glob(f"{file_path}/*.off")

#     means = []
#     stds = []

#     for f in test_files:
#         class_map[i] = f.split("/")[-1]
#         point_cloud = trimesh.load(f).sample(num_points)

#         mean = np.mean(point_cloud, axis=0)
#         std = np.std(point_cloud, axis=0)

#         normalized_point_cloud = (point_cloud - mean) / std

#         test_points.append(normalized_point_cloud)
#         test_labels.append(i)

#         means.append(mean)
#         stds.append(std)
#         i += 1
        
#     means = np.array(means)
#     stds = np.array(stds)

#     return (
#         np.array(test_points),
#         np.array(test_labels),
#         class_map
#     )

# def 
# model = keras.models.load_model('my_model.h5')

# for k in range(5):
def detect_foot(fpath):
    
    test_points1, test_labels1, CLASS_MAP1 = parse_unknown_dataset(fpath, NUM_POINTS)

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
            if CLASS_MAP1[labels.numpy()[i]].split(".")[0][-4:] == 'foot':
                global count
                count = count + 1
            else:
                print("__________incorrect__________)")
        elif f[i]==f.min():
            print("\tpred: {:}, \tlabel: {}, \tMIN".format(f[i], CLASS_MAP1[labels.numpy()[i]]))
        else:
            print("\tpred: {:}, \tlabel: {}".format(f[i], CLASS_MAP1[labels.numpy()[i]]))


# fpaths = glob.glob('test-files/*')
# global count
# count = 0
# for i in fpaths:
#     print()
#     print(i)
#     detect_foot(i)
    
# print(len(fpaths), count)

# from build_model import build_model
# model = build_model()
# print(model.summary())