from abc import ABC, abstractmethod
import tensorflow as tf


class Model(ABC):

    def __init__(self, obs_size, action_size, lr):

        self.obs_size = obs_size
        self.action_size = action_size
        self.lr = lr
    
    @abstractmethod
    def get_model(self) -> tf.keras.models: pass

class NaiveDQN(Model):

    def get_model(self):
        
        model = tf.keras.models.Sequential([
                    tf.keras.layers.Dense(128, input_shape=(self.obs_size,)),
                    tf.keras.layers.Activation('relu'),
                    tf.keras.layers.Dense(128),
                    tf.keras.layers.Activation('relu'),
                    tf.keras.layers.Dense(self.action_size)
            ])

        model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.lr), loss='mse')

        return model

        