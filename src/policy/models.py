from abc import ABC, abstractmethod
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers,models
import tensorflow_addons as tfa


class Model(ABC):

    def __init__(self, obs_size, action_size, lr, noisy):

        self.obs_size = obs_size
        self.action_size = action_size
        self.lr = lr
        self.noisy = noisy
    
    @abstractmethod
    def get_model(self) -> tf.keras.models: pass

    @abstractmethod
    def get_compiled_model(self) -> tf.keras.models: pass

class NaiveQNetwork(Model):

    def get_model(self):
        
        # model = models.Sequential([
        #             layers.Dense(128, input_shape=(self.obs_size,)),
        #             layers.Activation('relu'),
        #             layers.Dense(128),
        #             layers.Activation('relu'),
        #             layers.Dense(self.action_size)
        #     ])

        model = models.Sequential()
        model.add(keras.Input(shape=(self.obs_size,)))
        if self.noisy :
            model.add(tfa.layers.NoisyDense(128,activation='relu'))
        model.add(layers.Dense(128,activation='relu'))
        model.add(layers.Dense(128,activation='relu'))

        return model
    
    def get_compiled_model(self):
        model = self.get_model()
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.lr), loss='mse')

        return model 

class DuelingQNetwork(Model):

    def get_model(self):

        input = layers.Input(shape=(self.obs_size,))
        value = layers.Dense(128, activation="relu")(input)
        value = layers.Dense(128, activation="relu")(value)
        value = layers.Dense(1, activation="relu")(value)
        advantage = layers.Dense(128, activation="relu")(input)
        advantage = layers.Dense(128, activation="relu")(advantage)
        advantage = layers.Dense(self.action_size, activation="relu")(advantage)
        advantage_mean = layers.Lambda(lambda x: K.mean(x, axis=1))(advantage)
        advantage = layers.Subtract()([advantage, advantage_mean])
        out = layers.Add()([value, advantage]) 

        model = models.Model(inputs=input, outputs=out)

        return model

    def get_compiled_model(self):
        model = self.get_model()
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.lr), loss='mse')

        return model 


class DuelingQNetwork_2(Model):

    def get_model(self):

        input = layers.Input(shape=(self.obs_size,))
        if self.noisy :
            common_dense1 = tfa.layers.NoisyDense(128, activation="relu")(input)
        else :
            common_dense1 = layers.Dense(128, activation="relu")(input)
        common_dense2 = layers.Dense(1024, activation='relu', kernel_initializer='he_uniform')(common_dense1)
        common_dense3 = layers.Dense(512, activation='relu', kernel_initializer='he_uniform')(common_dense2)

        # value layer
        value = layers.Dense(1, activation='linear')(common_dense3)  
        # advantage layer
        advantage = layers.Dense(self.action_size, activation='linear')(common_dense3)  
        advantage = layers.Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True))(advantage)  

        # out layer (Value + Advantage)
        out = layers.Add()([value, advantage])  

        model = models.Model(inputs=[input], outputs=[out])
        return model


    def get_compiled_model(self):
        model = self.get_model()
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.lr), loss='mse')

        return model 
        

class NoisyQNetwork(Model):

    def get_model(self) -> tf.keras.models:
        return super().get_model()
        