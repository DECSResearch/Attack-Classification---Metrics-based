import tensorflow as tf
import keras as ke
from keras import Sequential
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Bidirectional
from openfl.federated import KerasTaskRunner

@ke.utils.register_keras_serializable()
class FedProxOptimizer(ke.optimizers.legacy.Optimizer):
    
    def __init__(self, mu=0.01, learning_rate=0.001, name="FedProxOptimizer", **kwargs):
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", learning_rate)
        self._set_hyper("mu", mu)
        self.vstars = {}
        
    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "vstar", initializer="zeros")
            
    def _prepare_local(self, var_device, var_dtype, apply_state):
        super()._prepare_local(var_device, var_dtype, apply_state)
        lr_t = tf.convert_to_tensor(self._get_hyper("learning_rate"))
        mu_t = tf.convert_to_tensor(self._get_hyper("mu"))
        
        apply_state[(var_device, var_dtype)].update(
            dict(
                lr_t=tf.cast(lr_t, var_dtype),
                mu_t=tf.cast(mu_t, var_dtype),
            )
        )
        
    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype)) or 
                       self._fallback_apply_state(var_device, var_dtype))
        
        vstar = self.get_slot(var, "vstar")
        lr_t = coefficients["lr_t"]
        mu_t = coefficients["mu_t"]
        
        var_update = var.assign_sub(
            lr_t * (grad + mu_t * (var - vstar))
        )
        
        return var_update
        
    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        raise NotImplementedError("Sparse updates not supported.")
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "mu": self._serialize_hyperparameter("mu"),
        })
        return config
    
class KerasLSTM(KerasTaskRunner):

    def __init__(self, input_shape, **kwargs):
        """
        Initialize.
        Args:
            input_shape (tuple): Shape of the input data (timesteps, features)
            **kwargs: Additional parameters to pass to the function
        """
        # Limit TensorFlow memory usage to 1 GB (adjust as needed)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])  # 1 GB
            except RuntimeError as e:
                print(e)
        

        super().__init__(**kwargs)
        self.input_shape = input_shape
        self.model = self.build_model(input_shape, **kwargs)
        self.initialize_tensorkeys_for_functions()
        self.model.summary(print_fn=self.logger.info)
        self.logger.info(f'Train Set Size : {self.get_train_data_size()}')
        self.logger.info(f'Valid Set Size : {self.get_valid_data_size()}')

    def build_model(self, input_shape, lstm_units=64, latent_dim=32, **kwargs):
        """Build and compile the model."""
        input_shape = (20, 4)
        
        model = Sequential()
        
        #LSTM
        model.add(LSTM(128, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, input_shape=input_shape))
        model.add(LSTM(64, activation='tanh', recurrent_activation='sigmoid', return_sequences=True))
        model.add(LSTM(32, activation='tanh', recurrent_activation='sigmoid', return_sequences=False))
        model.add(RepeatVector(input_shape[0]))
        model.add(LSTM(32, activation='tanh', recurrent_activation='sigmoid', return_sequences=True))
        model.add(LSTM(64, activation='tanh', recurrent_activation='sigmoid', return_sequences=True))
        model.add(LSTM(128, activation='tanh', recurrent_activation='sigmoid', return_sequences=True))
        model.add(TimeDistributed(Dense(input_shape[1])))
        
        #BiLSTM
        #model.add(Bidirectional(LSTM(64, activation='tanh', recurrent_activation='sigmoid', return_sequences=True), input_shape=input_shape))
        #model.add(Bidirectional(LSTM(32, activation='tanh', recurrent_activation='sigmoid', return_sequences=False)))
        #model.add(RepeatVector(input_shape[0]))
        #model.add(Bidirectional(LSTM(32, activation='tanh', recurrent_activation='sigmoid', return_sequences=True)))
        #model.add(Bidirectional(LSTM(64, activation='tanh', recurrent_activation='sigmoid', return_sequences=True)))
        #model.add(TimeDistributed(Dense(input_shape[1])))
        
        model.compile(
        
            #optimizer = FedProxOptimizer(learning_rate=0.001, mu=0.01), #FedProx
            optimizer=ke.optimizers.legacy.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7), #FedAvg
            loss='mae',
            metrics=['RootMeanSquaredError']
        )
        
        model.summary()
        print("\nModel built successfully")

        return model