import keras
from keras import layers
import tensorflow as tf

print(keras.__version__)

def conv_bn(x, filters):
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)

def dense_bn(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)


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


def tnet(inputs, num_features):
    # Initialise bias as the identity matrix
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)

    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = dense_bn(x, 128)
    x = layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=reg,
    )(x)
    feat_T = layers.Reshape((num_features, num_features))(x)
    # Apply affine transformation to input features
    return layers.Dot(axes=(2, 1))([inputs, feat_T])


model = keras.models.load_model('my_model.keras', custom_objects={'OrthogonalRegularizer': OrthogonalRegularizer})

print(model.summary())