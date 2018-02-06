import tensorflow as tf

class model():
    def __init__(self):
        pass

    def get_loss(self, features):
        return tf.Variable(0, dtype=tf.float32)

    def get_training_func(self, initializer):
        return self.get_loss