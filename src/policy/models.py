from abc import ABC, abstractmethod

import tensorflow as tf
import numpy as np 
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers,models
import tensorflow_addons as tfa
import tensorflow_probability as tfp

class CustomModel(ABC):

    def __init__(self, obs_shape, action_size, lr, noisy):

        self.obs_shape = obs_shape
        self.action_size = action_size
        self.lr = lr
        self.noisy = noisy

        self.obs_size = np.prod(self.obs_shape)
    
    @abstractmethod
    def get_model(self) -> tf.keras.models: pass

    @abstractmethod
    def get_compiled_model(self) -> tf.keras.models: pass

class NaiveQNetwork(CustomModel):

    def get_model(self):

        model = models.Sequential()
        model.add(layers.InputLayer(input_shape=(self.obs_size,)))
        if self.noisy :
            model.add(tfa.layers.NoisyDense(128,activation='relu'))
        model.add(layers.Dense(128,activation='relu'))
        model.add(layers.Dense(128,activation='relu'))
        model.add(layers.Dense(self.action_size))
        return model
    
    def get_compiled_model(self):
        model = self.get_model()
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.lr), loss='mse')

        return model 

class DuelingQNetwork(CustomModel):

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


class DuelingQNetwork_2(CustomModel):

    def get_model(self):
        
        input = layers.Input(shape=(self.obs_size,))
        if self.noisy :
            common_dense1 = tfa.layers.NoisyDense(128, activation="relu")(input)                         #noisy layer 
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


class PPOModel(keras.Model):

    def __init__(self, obs_shape, action_size):
        super().__init__()
        obs_size = np.prod(obs_shape)
        
        self.inputx = keras.layers.Dense(obs_size)
        self.dense1 = keras.layers.Dense(64, activation='relu', kernel_initializer=keras.initializers.he_normal())
        self.dense2 = keras.layers.Dense(64, activation='relu', kernel_initializer=keras.initializers.he_normal())
        self.value = keras.layers.Dense(1)
        self.policy_logits = keras.layers.Dense(action_size)

    def call(self, inputs):
        x = self.inputx(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.value(x), self.policy_logits(x)

    def action_value(self, obs):
        value, logits = self.predict_on_batch(obs)
        dist = tfp.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, value


class DuelingCNN(CustomModel):

    def get_model(self):
        
        #flattened array in input 
        input = layers.Input(shape=(self.obs_size,))

        #reshape as (height,width,depth)
        reshaped_input = layers.Reshape(self.obs_shape)(input)

        #convolutional layers
        conv1 = layers.Conv2D(32,3,padding='same',activation='relu')(reshaped_input)
        conv2 = layers.Conv2D(64,3,padding='same',activation='relu')(conv1)
        conv3 = layers.Conv2D(128,3,strides=(2,2),padding='same',activation='relu')(conv2)

        #dropout layer
        drop = layers.Dropout(0.5)(conv3)

        #flatten before dense aggregating layes
        flatten = layers.Flatten()(drop)
        
        #dense layers 
        common_dense1 = layers.Dense(128,activation='relu')(flatten)
        common_dense2 = layers.Dense(512, activation='relu')(common_dense1)
        common_dense3 = layers.Dense(256, activation='relu')(common_dense2)

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


