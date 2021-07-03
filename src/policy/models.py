from abc import ABC, abstractmethod
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers


class Model(ABC):

    def __init__(self, obs_size, action_size, lr):

        self.obs_size = obs_size
        self.action_size = action_size
        self.lr = lr
    
    @abstractmethod
    def get_model(self) -> tf.keras.models: pass

    @abstractmethod
    def get_compiled_model(self) -> tf.keras.models: pass

class NaiveQNetwork(Model):

    def get_model(self):
        
        model = keras.models.Sequential([
                    keras.layers.Dense(128, input_shape=(self.obs_size,)),
                    keras.layers.Activation('relu'),
                    keras.layers.Dense(128),
                    keras.layers.Activation('relu'),
                    keras.layers.Dense(self.action_size)
            ])

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

        model = keras.models.Model(inputs=input, outputs=out)

        return model

    def get_compiled_model(self):
        model = self.get_model()
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.lr), loss='mse')

        return model 


class DuelingQNetwork_beta(Model):

    def get_model(self):

        # input = layers.Input(shape=(self.obs_size,))
        # common_dense1 = layers.Dense(128, activation="relu")(input)
        # common_dense2 = layers.Dense(128, activation="relu")(common_dense1)
        # value = layers.Dense(128, activation="relu")(common_dense2)
        # value = layers.Dense(1, activation="relu")(value)
        # advantage = layers.Dense(128, activation="relu")(common_dense2)
        # advantage = layers.Dense(self.action_size, activation="relu")(advantage)
        # advantage_mean = layers.Lambda(lambda x: K.mean(x, axis=1))(advantage)
        # advantage = layers.Subtract()([advantage, advantage_mean])
        # out = layers.Add()([value, advantage]) 

        # model = keras.models.Model(inputs=input, outputs=out)

        input = layers.Input(shape=(self.obs_size,))

        
        X_layer = layers.Dense(128, activation="relu")(input)
        # action = layers.Dense(self._action_size, activation="linear")(X_layer) #TODO Parametrize layer sizes
        X_layer = layers.Dense(1024, activation='relu', kernel_initializer='he_uniform')(X_layer)
        X_layer = layers.Dense(512, activation='relu', kernel_initializer='he_uniform')(X_layer)

        # value layer
        V_layer = layers.Dense(1, activation='linear', name='V')(X_layer)  # V(S)
        # advantage layer
        A_layer = layers.Dense(self.action_size, activation='linear', name='Ai')(X_layer)  # A(s,a)
        A_layer = layers.Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), name='Ao')(A_layer)  # A(s,a)
        # Q layer (V + A)
        Q = layers.Add(name='Q')([V_layer, A_layer])  # Q(s,a)
        Q_model = keras.models.Model(inputs=[input], outputs=[Q], name='qvalue')
        return Q_model


    def get_compiled_model(self):
        model = self.get_model()
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.lr), loss='mse')

        return model 
        

class NoisyQNetwork(Model):

    def get_model(self) -> tf.keras.models:
        return super().get_model()
        