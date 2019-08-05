import os
import numpy as np
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
os.environ[
    "TF_CPP_MIN_LOG_LEVEL"
] = "3"  # Suppress UserWarning of TensorFlow while loading the model

from datetime import datetime
from functools import wraps
import shutil, zipfile, tempfile, pickle

import keras
from keras.layers import (
    Input,
    Concatenate,
    Dense,
    Flatten,
    RepeatVector,
    TimeDistributed,
    Bidirectional,
    GaussianNoise,
    BatchNormalization
)
from keras.layers import (
    CuDNNLSTM as LSTM,
)  # Faster drop-in for LSTM using CuDNN on TF backend on GPU
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from keras.utils import multi_gpu_model, plot_model

from sklearn.preprocessing import StandardScaler  # For the descriptors
from sklearn.decomposition import PCA  # For the descriptors

# Custom dependencies
from molvecgen import SmilesVectorizer
from .generators import CodeGenerator as DescriptorGenerator
from .generators import HetSmilesGenerator
from .custom_callbacks import ModelAndHistoryCheckpoint, LearningRateSchedule


def timed(func):
    """Timer decorator to benchmark functions."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        tstart = datetime.now()
        result = func(*args, **kwargs)
        elapsed = (datetime.now() - tstart).microseconds / 1e6
        print("Elapsed time: %.3f seconds." % elapsed)
        return result

    return wrapper


class DDC:
    def __init__(self, **kwargs):
        """Initialize a DDC object from scratch or from a trained configuration. All binary mols are converted to SMILES strings internally, using the vectorizers.
        
        # Examples of __init__ usage
            To *train* a blank model with encoder (autoencoder):
                model = ddc.DDC(x              = mols,
                                y              = mols,
                                scaling        = True,
                                pca            = True,
                                dataset_info   = info,
                                noise_std      = 0.1,
                                lstm_dim       = 256,
                                dec_layers     = 3,
                                td_dense_dim   = 0,
                                batch_size     = 128,
                                codelayer_dim  = 128)
            
            To *train* a blank model without encoder:
                model = ddc.DDC(x              = descriptors,
                                y              = mols,
                                scaling        = True,
                                pca            = True,
                                dataset_info   = info,
                                noise_std      = 0.1,
                                lstm_dim       = 256,
                                dec_layers     = 3,
                                td_dense_dim   = 0,
                                batch_size     = 128)
                                
            To *re-train* a saved model with encoder (autoencoder):
                model = ddc.DDC(x              = mols,
                                y              = mols,
                                model_name     = saved_model_name)
            
            To *re-train* a saved model without encoder:
                model = ddc.DDC(x              = descriptors,
                                y              = mols,
                                model_name     = saved_model_name)
                
            To *test* a saved model:
                model = ddc.DDC(model_name     = saved_model_name)

        :param x: Encoder input
        :type x: list or numpy.ndarray
        :param y: Decoder input for teacher's forcing
        :type y: list or numpy.ndarray
        :param scaling: Flag to scale descriptor inputs, defaults to `False`
        :type scaling: boolean
        :param pca: Flag to apply PCA on descriptor inputs, defaults to `False`
        :type pca: boolean
        :param model_name: Filename of model to load
        :type model_name: str
        :param dataset_info: Metadata about dataset
        :type dataset_info: dict
        :param noise_std: Standard deviation of noise in the latent space, defaults to 0.01
        :type noise_std: float
        :param lstm_dim: Number of LSTM units in the encoder/decoder layers, defaults to 256
        :param dec_layers: Number of decoder layers, defaults to 2
        :type dec_layers: int
        :param td_dense_dim: Number of intermediate Dense units to squeeze LSTM outputs, defaults to 0
        :type td_dense_dim: int
        :param batch_size: Batch size to train with, defaults to 256
        :type batch_size: int
        :param codelayer_dim: Dimensionality of latent space
        :type codelayer_dim: int
        :param bn: Fla to enable batch normalization, defaults to `True`
        :type bn: boolean
        :param bn_momentum: Momentum value to be used in batch normalization, defaults to 0.9
        :type bn_momentum: float
        """

        # Identify the mode to start the model in
        if "x" in kwargs and "y" in kwargs:
            x = kwargs.get("x")
            y = kwargs.get("y")
            if "model_name" not in kwargs:
                self.__mode = "train"
            else:
                self.__mode = "retrain"
        elif "model_name" in kwargs:
            self.__mode = "test"
        else:
            raise NameError("Cannot infer mode from arguments.")

        print("Initializing model in %s mode." % self.__mode)

        if self.mode == "train":
            # Infer input type from type(x)
            if type(x[0]) == np.bytes_:
                print("Input type is 'binary mols'.")
                self.__input_type = "mols"  # binary RDKit mols
            else:
                print("Input type is 'molecular descriptors'.")
                self.__input_type = "descriptors"  # other molecular descriptors

                # If scaling is required
                if kwargs.get("scaling", False) is True:
                    # Normalize the input
                    print("Applying scaling on input.")
                    self.__scaler = StandardScaler()
                    x = self.__scaler.fit_transform(x)
                else:
                    self.__scaler = None

                # If PCA is required
                if kwargs.get("pca", False) is True:
                    print("Applying PCA on input.")
                    self.__pca = PCA(
                        n_components=x.shape[1]
                    )  # n_components=n_features for now
                    x = self.__pca.fit_transform(x)
                else:
                    self.__pca = None

            self.__maxlen = (
                    kwargs.get("dataset_info")["maxlen"] + 10
            )  # Extend maxlen to avoid breaks in training
            self.__charset = kwargs.get("dataset_info")["charset"]
            self.__dataset_name = kwargs.get("dataset_info")["name"]
            self.__lstm_dim = kwargs.get("lstm_dim", 256)
            self.__h_activation = kwargs.get("h_activation", "relu")
            self.__bn = kwargs.get("bn", True)
            self.__bn_momentum = kwargs.get("bn_momentum", 0.9)
            self.__noise_std = kwargs.get("noise_std", 0.01)
            self.__td_dense_dim = kwargs.get(
                "td_dense_dim", 0
            )  # >0 squeezes RNN connections with Dense sandwiches
            self.__batch_size = kwargs.get("batch_size", 256)
            self.__dec_layers = kwargs.get("dec_layers", 2)

            if self.input_type == "descriptors":
                self.__codelayer_dim = x.shape[1]  # features
                if "codelayer_dim" in kwargs:
                    print(
                        "Ignoring requested codelayer_dim because it is inferred from the cardinality of the descriptors."
                    )
            else:
                self.__codelayer_dim = kwargs.get("codelayer_dim", 128)

            # Create the left/right-padding vectorizers
            self.__smilesvec1 = SmilesVectorizer(
                canonical=False,
                augment=True,
                maxlength=self.maxlen,
                charset=self.charset,
                binary=True,
            )

            self.__smilesvec2 = SmilesVectorizer(
                canonical=False,
                augment=True,
                maxlength=self.maxlen,
                charset=self.charset,
                binary=True,
                leftpad=False,
            )

            # self.train_gen.next() #This line is needed to set train_gen.dims (to be fixed in HetSmilesGenerator)
            self.__input_shape = self.smilesvec1.dims
            self.__dec_dims = list(self.smilesvec1.dims)
            self.__dec_dims[0] = self.dec_dims[0] - 1
            self.__dec_input_shape = self.dec_dims
            self.__output_len = self.smilesvec1.dims[0] - 1
            self.__output_dims = self.smilesvec1.dims[-1]

            # Build all sub-models as untrained models
            if self.input_type == "mols":
                self.__build_mol_to_latent_model()
            else:
                self.__mol_to_latent_model = None

            self.__build_latent_to_states_model()
            self.__build_batch_model()

            # Build data generators
            self.__build_generators(x, y)

        # Retrain or Test mode
        else:
            self.__model_name = kwargs.get("model_name")

            # Load the model
            self.__load(self.model_name)

            if self.mode == "retrain":
                # Build data generators
                self.__build_generators(x, y)

        # Build full model out of the sub-models
        self.__build_model()

        # Show the resulting full model
        print(self.model.summary())

    """
    Architecture properties.
    """

    @property
    def lstm_dim(self):
        return self.__lstm_dim

    @property
    def h_activation(self):
        return self.__h_activation

    @property
    def bn(self):
        return self.__bn

    @property
    def bn_momentum(self):
        return self.__bn_momentum

    @property
    def noise_std(self):
        return self.__noise_std

    @property
    def td_dense_dim(self):
        return self.__td_dense_dim

    @property
    def batch_size(self):
        return self.__batch_size

    @property
    def dec_layers(self):
        return self.__dec_layers

    @property
    def codelayer_dim(self):
        return self.__codelayer_dim

    @property
    def steps_per_epoch(self):
        return self.__steps_per_epoch

    @property
    def validation_steps(self):
        return self.__validation_steps

    @property
    def input_shape(self):
        return self.__input_shape

    @property
    def dec_dims(self):
        return self.__dec_dims

    @property
    def dec_input_shape(self):
        return self.__dec_input_shape

    @property
    def output_len(self):
        return self.__output_len

    @property
    def output_dims(self):
        return self.__output_dims

    @property
    def batch_input_length(self):
        return self.__batch_input_length

    @batch_input_length.setter
    def batch_input_length(self, value):
        self.__batch_input_length = value
        self.__build_sample_model(batch_input_length=value)

    """
    Models.
    """

    @property
    def mol_to_latent_model(self):
        return self.__mol_to_latent_model

    @property
    def latent_to_states_model(self):
        return self.__latent_to_states_model

    @property
    def batch_model(self):
        return self.__batch_model

    @property
    def sample_model(self):
        return self.__sample_model

    @property
    def multi_sample_model(self):
        return self.__multi_sample_model

    @property
    def model(self):
        return self.__model

    """
    Train properties.
    """

    @property
    def epochs(self):
        return self.__epochs

    @property
    def clipvalue(self):
        return self.__clipvalue

    @property
    def lr(self):
        return self.__lr

    @property
    def h(self):
        return self.__h

    @h.setter
    def h(self, value):
        self.__h = value

    """
    Other properties.
    """

    @property
    def mode(self):
        return self.__mode

    @property
    def dataset_name(self):
        return self.__dataset_name

    @property
    def model_name(self):
        return self.__model_name

    @property
    def input_type(self):
        return self.__input_type

    @property
    def maxlen(self):
        return self.__maxlen

    @property
    def charset(self):
        return self.__charset

    @property
    def smilesvec1(self):
        return self.__smilesvec1

    @property
    def smilesvec2(self):
        return self.__smilesvec2

    @property
    def train_gen(self):
        return self.__train_gen

    @property
    def valid_gen(self):
        return self.__valid_gen

    @property
    def scaler(self):
        try:
            return self.__scaler
        except:
            return None

    @property
    def pca(self):
        try:
            return self.__pca
        except:
            return None

    """
    Private methods.
    """

    def __build_generators(self, x, y, split=0.9):
        """Build data generators to be used for (re)training.
        
        :param x: Encoder input
        :type x: list
        :param y: Decoder input for teacher's forcing
        :type y: list
        :param split: Fraction of samples to keep for training (rest for validation), defaults to 0.9
        :type split: float, optional
        """

        # Sanity check
        assert len(x) == len(y)

        # Split dataset into train and validation sets
        cut = int(split * len(x))
        x_train = x[:cut]
        x_valid = x[cut:]
        y_train = y[:cut]
        y_valid = y[cut:]

        if self.input_type == "mols":
            self.__train_gen = HetSmilesGenerator(
                x_train,
                None,
                self.smilesvec1,
                self.smilesvec2,
                batch_size=self.batch_size,
                shuffle=True,
            )

            self.__valid_gen = HetSmilesGenerator(
                x_valid,
                None,
                self.smilesvec1,
                self.smilesvec2,
                batch_size=self.batch_size,
                shuffle=True,
            )

        else:
            self.__train_gen = DescriptorGenerator(
                x_train,
                y_train,
                self.smilesvec1,
                self.smilesvec2,
                batch_size=self.batch_size,
                shuffle=True,
            )

            self.__valid_gen = DescriptorGenerator(
                x_valid,
                y_valid,
                self.smilesvec1,
                self.smilesvec2,
                batch_size=self.batch_size,
                shuffle=True,
            )

        # Calculate number of batches per training/validation epoch
        train_samples = len(x_train)
        valid_samples = len(x_valid)
        self.__steps_per_epoch = train_samples // self.batch_size
        self.__validation_steps = valid_samples // self.batch_size

        print(
            "Model received %d train samples and %d validation samples."
            % (train_samples, valid_samples)
        )

    def __build_mol_to_latent_model(self):
        """Model that transforms binary molecules to their latent representation.
        Only used if input is mols.
        """

        # Input tensor (MANDATORY)
        encoder_inputs = Input(shape=self.input_shape, name="Encoder_Inputs")

        x = encoder_inputs

        # The two encoder layers, number of cells are halved as Bidirectional
        encoder = Bidirectional(
            LSTM(
                self.lstm_dim // 2,
                return_sequences=True,
                return_state=True,  # Return the states at end of the batch
                name="Encoder_LSTM_1",
            )
        )

        x, state_h, state_c, state_h_reverse, state_c_reverse = encoder(x)

        if self.bn:
            x = BatchNormalization(momentum=self.bn_momentum, name="BN_1")(x)

        encoder2 = Bidirectional(
            LSTM(
                self.lstm_dim // 2,
                return_state=True,  # Return the states at end of the batch
                name="Encoder_LSTM_2",
            )
        )

        _, state_h2, state_c2, state_h2_reverse, state_c2_reverse = encoder2(x)

        # Concatenate all states of the forward and the backward LSTM layers
        states = Concatenate(axis=-1, name="Concatenate_1")(
            [
                state_h,
                state_c,
                state_h2,
                state_c2,
                state_h_reverse,
                state_c_reverse,
                state_h2_reverse,
                state_c2_reverse,
            ]
        )

        if self.bn:
            states = BatchNormalization(momentum=self.bn_momentum, name="BN_2")(states)

        # A non-linear recombination
        neck_relu = Dense(
            self.codelayer_dim, activation=self.h_activation, name="Codelayer_Relu"
        )
        neck_outputs = neck_relu(states)

        if self.bn:
            neck_outputs = BatchNormalization(
                momentum=self.bn_momentum, name="BN_Codelayer"
            )(neck_outputs)

        # Add Gaussian noise to "spread" the distribution of the latent variables during training
        neck_outputs = GaussianNoise(self.noise_std, name="Gaussian_Noise")(
            neck_outputs
        )

        # Define the model
        self.__mol_to_latent_model = Model(encoder_inputs, neck_outputs)

        # Name it!
        self.mol_to_latent_model.name = "mol_to_latent_model"

    def __build_latent_to_states_model(self):
        """Model that constructs the initial states of the decoder from a latent molecular representation.
        """

        # Input tensor (MANDATORY)
        latent_input = Input(shape=(self.codelayer_dim,), name="Latent_Input")

        # Initialize list of state tensors for the decoder
        decoder_state_list = []

        for dec_layer in range(self.dec_layers):

            # The tensors for the initial states of the decoder
            name = "Dense_h_" + str(dec_layer)
            h_decoder = Dense(self.lstm_dim, activation="relu", name=name)(latent_input)

            name = "Dense_c_" + str(dec_layer)
            c_decoder = Dense(self.lstm_dim, activation="relu", name=name)(latent_input)

            if self.bn:
                name = "BN_h_" + str(dec_layer)
                h_decoder = BatchNormalization(momentum=self.bn_momentum, name=name)(
                    h_decoder
                )

                name = "BN_c_" + str(dec_layer)
                c_decoder = BatchNormalization(momentum=self.bn_momentum, name=name)(
                    c_decoder
                )

            decoder_state_list.append(h_decoder)
            decoder_state_list.append(c_decoder)

        # Define the model
        self.__latent_to_states_model = Model(latent_input, decoder_state_list)

        # Name it!
        self.latent_to_states_model.name = "latent_to_states_model"

    def __build_batch_model(self):
        """Model that returns a vectorized SMILES string of OHE characters.
        """

        # List of input tensors to batch_model
        inputs = []

        # This is the start character padded OHE smiles for teacher forcing
        decoder_inputs = Input(shape=self.dec_input_shape, name="Decoder_Inputs")
        inputs.append(decoder_inputs)

        # I/O tensor of the LSTM layers
        x = decoder_inputs

        for dec_layer in range(self.dec_layers):
            name = "Decoder_State_h_" + str(dec_layer)
            state_h = Input(shape=[self.lstm_dim], name=name)
            inputs.append(state_h)

            name = "Decoder_State_c_" + str(dec_layer)
            state_c = Input(shape=[self.lstm_dim], name=name)
            inputs.append(state_c)

            # RNN layer
            decoder_lstm = LSTM(
                self.lstm_dim,
                return_sequences=True,
                name="Decoder_LSTM_" + str(dec_layer),
            )

            x = decoder_lstm(x, initial_state=[state_h, state_c])

            if self.bn:
                x = BatchNormalization(
                    momentum=self.bn_momentum, name="BN_Decoder_" + str(dec_layer)
                )(x)

            # Squeeze LSTM interconnections using Dense layers
            if self.td_dense_dim > 0:
                x = TimeDistributed(
                    Dense(self.td_dense_dim), name="Time_Distributed_" + str(dec_layer)
                )(x)

        # Final Dense layer to return soft labels (probabilities)
        outputs = Dense(self.output_dims, activation="softmax", name="Dense_Decoder")(x)

        # Define the batch_model
        self.__batch_model = Model(inputs=inputs, outputs=[outputs])

        # Name it!
        self.batch_model.name = "batch_model"

    def __build_model(self):
        """Full model that constitutes the complete pipeline.
        """

        # IFF input is not encoded, stack the encoder (mol_to_latent_model)
        if self.input_type == "mols":
            # Input tensor (MANDATORY) - Same as the mol_to_latent_model input!
            encoder_inputs = Input(shape=self.input_shape, name="Encoder_Inputs")
            # Input tensor (MANDATORY) - Same as the batch_model input for teacher's forcing!
            decoder_inputs = Input(shape=self.dec_input_shape, name="Decoder_Inputs")

            # Stack the three models
            # Propagate tensors through 1st model
            x = self.mol_to_latent_model(encoder_inputs)
            # Propagate tensors through 2nd model
            x = self.latent_to_states_model(x)
            # Append the first input of the third model to be the one for teacher's forcing
            x = [decoder_inputs] + x
            # Propagate tensors through 3rd model
            x = self.batch_model(x)

            # Define full model (SMILES -> SMILES)
            self.__model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=[x])

        # Input is pre-encoded, no need for encoder
        else:
            # Input tensor (MANDATORY)
            latent_input = Input(shape=(self.codelayer_dim,), name="Latent_Input")
            # Input tensor (MANDATORY) - Same as the batch_model input for teacher's forcing!
            decoder_inputs = Input(shape=self.dec_input_shape, name="Decoder_Inputs")

            # Stack the two models
            # Propagate tensors through 1st model
            x = self.latent_to_states_model(latent_input)
            # Append the first input of the 2nd model to be the one for teacher's forcing
            x = [decoder_inputs] + x
            # Propagate tensors through 2nd model
            x = self.batch_model(x)

            # Define full model (latent -> SMILES)
            self.__model = Model(inputs=[latent_input, decoder_inputs], outputs=[x])

    def __build_sample_model(self, batch_input_length) -> dict:
        """Model that predicts a single OHE character.
        This model is generated from the modified config file of the batch_model.
        
        :param batch_input_length: Size of generated batch
        :type batch_input_length: int
        :return: The dictionary of the configuration
        :rtype: dict
        """

        self.__batch_input_length = batch_input_length

        # Get the configuration of the batch_model
        config = self.batch_model.get_config()

        # Keep only the "Decoder_Inputs" as single input to the sample_model
        config["input_layers"] = [config["input_layers"][0]]

        # Find decoder states that are used as inputs in batch_model and remove them
        idx_list = []
        for idx, layer in enumerate(config["layers"]):

            if "Decoder_State_" in layer["name"]:
                idx_list.append(idx)

        # Pop the layer from the layer list
        # Revert indices to avoid re-arranging after deleting elements
        for idx in sorted(idx_list, reverse=True):
            config["layers"].pop(idx)

        # Remove inbound_nodes dependencies of remaining layers on deleted ones
        for layer in config["layers"]:
            idx_list = []

            try:
                for idx, inbound_node in enumerate(layer["inbound_nodes"][0]):
                    if "Decoder_State_" in inbound_node[0]:
                        idx_list.append(idx)
            # Catch the exception for first layer (Decoder_Inputs) that has empty list of inbound_nodes[0]
            except:
                pass

            # Pop the inbound_nodes from the list
            # Revert indices to avoid re-arranging
            for idx in sorted(idx_list, reverse=True):
                layer["inbound_nodes"][0].pop(idx)

        # Change the batch_shape of input layer
        config["layers"][0]["config"]["batch_input_shape"] = (
            batch_input_length,
            1,
            self.dec_input_shape[-1],
        )

        # Finally, change the statefulness of the RNN layers
        for layer in config["layers"]:
            if "Decoder_LSTM_" in layer["name"]:
                layer["config"]["stateful"] = True
                # layer["config"]["return_sequences"] = True

        # Define the sample_model using the modified config file
        sample_model = Model.from_config(config)

        # Copy the trained weights from the trained batch_model to the untrained sample_model
        for layer in sample_model.layers:
            # Get weights from the batch_model
            weights = self.batch_model.get_layer(layer.name).get_weights()
            # Set the weights to the sample_model
            sample_model.get_layer(layer.name).set_weights(weights)

        if batch_input_length == 1:
            self.__sample_model = sample_model

        elif batch_input_length > 1:
            self.__multi_sample_model = sample_model

        return config

    def __load(self, model_name):
        """Load a DDC object from a zip file.
        
        :param model_name: Path to model
        :type model_name: string
        """

        print("Loading model.")
        tstart = datetime.now()

        # Temporary directory to extract the zipped information
        with tempfile.TemporaryDirectory() as dirpath:

            # Unzip the directory that contains the saved model(s)
            with zipfile.ZipFile(model_name + ".zip", "r") as zip_ref:
                zip_ref.extractall(dirpath)

            # Load metadata
            metadata = pickle.load(open(dirpath + "/metadata.pickle", "rb"))

            # Re-load metadata
            self.__dict__.update(metadata)

            # Load all sub-models
            try:
                self.__mol_to_latent_model = load_model(
                    dirpath + "/mol_to_latent_model.h5"
                )
            except:
                print("'mol_to_latent_model' not found, setting to None.")
                self.__mol_to_latent_model = None

            self.__latent_to_states_model = load_model(
                dirpath + "/latent_to_states_model.h5"
            )
            self.__batch_model = load_model(dirpath + "/batch_model.h5")

            # Build sample_model out of the trained batch_model
            self.__build_sample_model(batch_input_length=1)  # Single-output model
            self.__build_sample_model(
                batch_input_length=self.batch_size
            )  # Multi-output model

        print("Loading finished in %i seconds." % ((datetime.now() - tstart).seconds))

    """
    Public methods.
    """

    def fit(
            self,
            epochs,
            lr,
            mini_epochs,
            patience,
            model_name,
            gpus=1,
            workers=1,
            use_multiprocessing=False,
            verbose=2,
            max_queue_size=10,
            clipvalue=0,
            save_period=5,
            checkpoint_dir="/projects/cc/kjmv588/models/checkpoints/",
            lr_decay=False,
            sch_epoch_to_start=500,
            sch_last_epoch=999,
            sch_lr_init=1e-3,
            sch_lr_final=1e-6,
    ):
        """Fit the full model to the training data.
        Supports multi-gpu training if gpus set to >1.
        
        :param epochs: Training iterations over complete training set.
        :type epochs: int
        :param lr: Initial learning rate
        :type lr: float
        :param mini_epochs: Subdivisions of a single epoch to trick Keras into applying callbacks
        :type mini_epochs: int
        :param patience: minimum consecutive mini_epochs of stagnated learning rate to consider before lowering it with ReduceLROnPlateau 
        :type patience: int
        :param model_name: Base name of model checkpoints
        :type model_name: str
        :param gpus: Number of GPUs to be used for training, defaults to 1
        :type gpus: int, optional
        :param workers: Keras CPU workers, defaults to 1
        :type workers: int, optional
        :param use_multiprocessing: Multi-CPU processing, defaults to False
        :type use_multiprocessing: bool, optional
        :param verbose: Keras training verbosity, defaults to 2
        :type verbose: int, optional
        :param max_queue_size: Keras generator max number of fetched samples, defaults to 10
        :type max_queue_size: int, optional
        :param clipvalue: Gradient clipping value, defaults to 0
        :type clipvalue: int, optional
        :param save_period: Checkpoint period in miniepochs, defaults to 5
        :type save_period: int, optional
        :param checkpoint_dir: Directory to store checkpoints in, defaults to "/projects/cc/kjmv588/models/checkpoints/"
        :type checkpoint_dir: str, optional
        :param lr_decay: Flag to enable exponential learning rate decay, defaults to False
        :type lr_decay: bool, optional
        :param sch_epoch_to_start: Miniepoch to start exponential learning rate decay, defaults to 500
        :type sch_epoch_to_start: int, optional
        :param sch_last_epoch: Last miniepoch of exponential learning rate decay, defaults to 999
        :type sch_last_epoch: int, optional
        :param sch_lr_init: Initial learning rate to start exponential learning rate decay, defaults to 1e-3
        :type sch_lr_init: float, optional
        :param sch_lr_final: Target learning rate value to stop decaying, defaults to 1e-6
        :type sch_lr_final: float, optional
        """

        # Get parameter values if specified
        self.__epochs = epochs
        self.__lr = lr
        self.__clipvalue = clipvalue

        # Optimizer
        if clipvalue > 0:
            print("Using gradient clipping %.2f." % clipvalue)
            opt = Adam(lr=self.lr, clipvalue=self.clipvalue)

        else:
            opt = Adam(lr=self.lr)

        checkpoint_file = (
                checkpoint_dir + "%s--{epoch:02d}--{val_loss:.4f}--{lr:.7f}" % model_name
        )

        # If model is untrained, history is blank
        try:
            history = self.h

        # Else, append the history
        except:
            history = {}

        # Callback for saving intermediate models during training
        mhcp = ModelAndHistoryCheckpoint(
            filepath=checkpoint_file,
            model_dict=self.__dict__,
            monitor="val_loss",
            verbose=1,
            mode="min",
            period=save_period,
            history=history,
        )
        # Training history
        self.__h = mhcp.history

        if lr_decay:
            lr_schedule = LearningRateSchedule(
                epoch_to_start=sch_epoch_to_start,
                last_epoch=sch_last_epoch,
                lr_init=sch_lr_init,
                lr_final=sch_lr_final,
            )

            lr_scheduler = LearningRateScheduler(
                schedule=lr_schedule.exp_decay, verbose=1
            )

            callbacks = [lr_scheduler, mhcp]

        else:
            rlr = ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=patience,
                min_lr=1e-6,
                verbose=1,
                min_delta=1e-4,
            )

            callbacks = [rlr, mhcp]

        # Inspect training parameters at the start of the training
        self.summary()

        # Parallel training on multiple GPUs
        if gpus > 1:
            parallel_model = multi_gpu_model(self.model, gpus=gpus)
            parallel_model.compile(loss="categorical_crossentropy", optimizer=opt)
            # This `fit` call will be distributed on all GPUs.
            # Each GPU will process (batch_size/gpus) samples per batch.
            parallel_model.fit_generator(
                self.train_gen,
                steps_per_epoch=self.steps_per_epoch / mini_epochs,
                epochs=mini_epochs * self.epochs,
                validation_data=self.valid_gen,
                validation_steps=self.validation_steps / mini_epochs,
                callbacks=callbacks,
                max_queue_size=max_queue_size,
                workers=workers,
                use_multiprocessing=use_multiprocessing,
                verbose=verbose,
            )  # 1 to show progress bar

        elif gpus == 1:
            self.model.compile(loss="categorical_crossentropy", optimizer=opt)
            self.model.fit_generator(
                self.train_gen,
                steps_per_epoch=self.steps_per_epoch / mini_epochs,
                epochs=mini_epochs * self.epochs,
                validation_data=self.valid_gen,
                validation_steps=self.validation_steps / mini_epochs,
                callbacks=callbacks,
                max_queue_size=10,
                workers=workers,
                use_multiprocessing=use_multiprocessing,
                verbose=verbose,
            )  # 1 to show progress bar

        # Build sample_model out of the trained batch_model
        self.__build_sample_model(batch_input_length=1)  # Single-output model
        self.__build_sample_model(
            batch_input_length=self.batch_size
        )  # Multi-output model

    def vectorize(self, mols_test, leftpad=True):
        """Perform One-Hot Encoding (OHE) on a binary molecule.
        
        :param mols_test: Molecules to vectorize
        :type mols_test: list
        :param leftpad: Left zero-padding direction, defaults to True
        :type leftpad: bool, optional
        :return: One-Hot Encoded molecules
        :rtype: list
        """

        if leftpad:
            return self.smilesvec1.transform(mols_test)
        else:
            return self.smilesvec2.transform(mols_test)

    def transform(self, mols_ohe):
        """Encode a batch of OHE molecules into their latent representations.
        Must be called on the output of self.vectorize().
        
        :param mols_ohe: List of One-Hot Encoded molecules
        :type mols_ohe: list
        :return: Latent representation of input molecules
        :rtype: list
        """

        latent = self.mol_to_latent_model.predict(mols_ohe)
        return latent.reshape((latent.shape[0], 1, latent.shape[1]))

    # @timed
    def predict(self, latent, temp=1):
        """Generate a single SMILES string.
        The states of the RNN are set based on the latent input.
        Careful, "latent" must be: the output of self.transform()
                                   or
                                   an array of molecular descriptors.
        If temp>0, multinomial sampling is used instead of selecting 
        the single most probable character at each step.
        If temp==1, multinomial sampling without temperature scaling is used.
        
        :param latent: 1D Latent vector to steer the generation
        :type latent: numpy.ndarray
        :param temp: Temperatute of multinomial sampling (argmax if 0), defaults to 1
        :type temp: int, optional
        :return: The predicted SMILES string and its NLL of being sampled
        :rtype: list
        """

        # Scale inputs if model is trained on scaled data
        if self.scaler is not None:
            latent = self.scaler.transform(
                latent.reshape(1, -1)
            )  # Re-shape because scaler complains

        # Apply PCA to input if model is trained accordingly
        if self.pca is not None:
            latent = self.pca.transform(latent)

        states = self.latent_to_states_model.predict(latent)

        # Decode states and reset the LSTM cells with them to bias the generation towards the desired properties
        for dec_layer in range(self.dec_layers):
            self.sample_model.get_layer("Decoder_LSTM_" + str(dec_layer)).reset_states(
                states=[states[2 * dec_layer], states[2 * dec_layer + 1]]
            )

        # Prepare the input char
        startidx = self.smilesvec1._char_to_int[self.smilesvec1.startchar]
        samplevec = np.zeros((1, 1, self.smilesvec1.dims[-1]))
        samplevec[0, 0, startidx] = 1
        smiles = ""
        # Initialize Negative Log-Likelihood (NLL)
        NLL = 0
        # Loop and predict next char
        for i in range(1000):
            o = self.sample_model.predict(samplevec)
            # Multinomial sampling with temperature scaling
            if temp > 0:
                temp = abs(temp)  # Handle negative values
                nextCharProbs = np.log(o) / temp
                nextCharProbs = np.exp(nextCharProbs)
                nextCharProbs = (
                        nextCharProbs / nextCharProbs.sum() - 1e-8
                )  # Re-normalize for float64 to make exactly 1.0 for np.random.multinomial
                sampleidx = np.random.multinomial(
                    1, nextCharProbs.squeeze(), 1
                ).argmax()

            # Else, select the most probable character
            else:
                sampleidx = np.argmax(o)

            samplechar = self.smilesvec1._int_to_char[sampleidx]
            if samplechar != self.smilesvec1.endchar:
                # Append the new character
                smiles += samplechar
                samplevec = np.zeros((1, 1, self.smilesvec1.dims[-1]))
                samplevec[0, 0, sampleidx] = 1
                # Calculate negative log likelihood for the selected character given the sequence so far
                NLL -= np.log(o[0][0][sampleidx])
            else:
                return smiles, NLL

    # @timed
    def predict_batch(self, latent, temp=1):
        """Generate multiple biased SMILES strings.
        Careful, "latent" must be: the output of self.transform()
                                   or
                                   an array of molecular descriptors.
        If temp>0, multinomial sampling is used instead of selecting 
        the single most probable character at each step.
        If temp==1, multinomial sampling without temperature scaling is used.
        Low temp leads to elimination of characters with low probabilities.
        predict_many() generates batch_input_length (default==batch_size) individual SMILES 
        strings per call. To change that, reset batch_input_length to a new value.
        
        :param latent: List of latent vectors
        :type latent: list
        :param temp: Temperatute of multinomial sampling (argmax if 0), defaults to 1
        :type temp: int, optional
        :return: List of predicted SMILES strings and their NLL of being sampled
        :rtype: list
        """

        if latent.shape[0] == 1:
            # Make a batch prediction by repeating the same latent vector for every neuron
            latent = np.ones((self.batch_input_length, self.codelayer_dim)) * latent
        else:
            # Make sure it is squeezed
            latent = latent.squeeze()

        # Scale inputs if model is trained on scaled data
        if self.scaler is not None:
            latent = self.scaler.transform(latent)

        # Apply PCA to input if model is trained accordingly
        if self.pca is not None:
            latent = self.pca.transform(latent)

        # Decode states and reset the LSTM cells with them, to bias the generation towards the desired properties
        states = self.latent_to_states_model.predict(latent)

        for dec_layer in range(self.dec_layers):
            self.multi_sample_model.get_layer(
                "Decoder_LSTM_" + str(dec_layer)
            ).reset_states(states=[states[2 * dec_layer], states[2 * dec_layer + 1]])

        # Index of input char "^"
        startidx = self.smilesvec1._char_to_int[self.smilesvec1.startchar]
        # Vectorize the input char for all SMILES
        samplevec = np.zeros((self.batch_input_length, 1, self.smilesvec1.dims[-1]))
        samplevec[:, 0, startidx] = 1
        # Initialize arrays to store SMILES, their NLLs and their status
        smiles = np.array([""] * self.batch_input_length, dtype=object)
        NLL = np.zeros((self.batch_input_length,))
        finished = np.array([False] * self.batch_input_length)

        # Loop and predict next char
        for i in range(1000):
            o = self.multi_sample_model.predict(
                samplevec, batch_size=self.batch_input_length
            ).squeeze()

            # Multinomial sampling with temperature scaling
            if temp > 0:
                temp = abs(temp)  # No negative values
                nextCharProbs = np.log(o) / temp
                nextCharProbs = np.exp(nextCharProbs)  # .squeeze()

                # Normalize probabilities
                nextCharProbs = (nextCharProbs.T / nextCharProbs.sum(axis=1) - 1e-8).T
                sampleidc = np.asarray(
                    [
                        np.random.multinomial(1, nextCharProb, 1).argmax()
                        for nextCharProb in nextCharProbs
                    ]
                )

            else:
                sampleidc = np.argmax(o, axis=1)

            samplechars = [self.smilesvec1._int_to_char[idx] for idx in sampleidc]

            for idx, samplechar in enumerate(samplechars):
                if not finished[idx]:
                    if samplechar != self.smilesvec1.endchar:
                        # Append the SMILES with the next character
                        smiles[idx] += self.smilesvec1._int_to_char[sampleidc[idx]]
                        samplevec = np.zeros(
                            (self.batch_input_length, 1, self.smilesvec1.dims[-1])
                        )
                        # One-Hot Encode the character
                        # samplevec[:,0,sampleidc] = 1
                        for count, sampleidx in enumerate(sampleidc):
                            samplevec[count, 0, sampleidx] = 1
                        # Calculate negative log likelihood for the selected character given the sequence so far
                        NLL[idx] -= np.log(o[idx][sampleidc[idx]])
                    else:
                        finished[idx] = True
                        # print("SMILES has finished at %i" %i)

            # If all SMILES are finished, i.e. the endchar "$" has been generated, stop the generation
            if finished.sum() == len(finished):
                return smiles, NLL

    @timed
    def get_smiles_nll(self, latent, smiles_ref) -> float:
        """Back-calculate the NLL of a given SMILES string if its descriptors are used as RNN states.
        
        :param latent: Descriptors or latent representation of smiles_ref
        :type latent: list
        :param smiles_ref: Given SMILES to back-calculate its NLL
        :type smiles_ref: str
        :return: NLL of sampling smiles_ref given its latent representation (or descriptors)
        :rtype: float
        """

        # Scale inputs if model is trained on scaled data
        if self.scaler is not None:
            latent = self.scaler.transform(
                latent.reshape(1, -1)
            )  # Re-shape because scaler complains

        # Apply PCA to input if model is trained accordingly
        if self.pca is not None:
            latent = self.pca.transform(latent)

        states = self.latent_to_states_model.predict(latent)

        # Decode states and reset the LSTM cells with them to bias the generation towards the desired properties
        for dec_layer in range(self.dec_layers):
            self.sample_model.get_layer("Decoder_LSTM_" + str(dec_layer)).reset_states(
                states=[states[2 * dec_layer], states[2 * dec_layer + 1]]
            )

        # Prepare the input char
        startidx = self.smilesvec1._char_to_int[self.smilesvec1.startchar]
        samplevec = np.zeros((1, 1, self.smilesvec1.dims[-1]))
        samplevec[0, 0, startidx] = 1

        # Initialize Negative Log-Likelihood (NLL)
        NLL = 0
        # Loop and predict next char
        for i in range(1000):
            o = self.sample_model.predict(samplevec)

            samplechar = smiles_ref[i]
            sampleidx = self.smilesvec1._char_to_int[samplechar]

            if i != len(smiles_ref) - 1:
                samplevec = np.zeros((1, 1, self.smilesvec1.dims[-1]))
                samplevec[0, 0, sampleidx] = 1
                # Calculate negative log likelihood for the selected character given the sequence so far
                NLL -= np.log(o[0][0][sampleidx])
            else:
                return NLL

    @timed
    def get_smiles_nll_batch(self, latent, smiles_ref) -> list:
        """Back-calculate the individual NLL for a batch of known SMILES strings.
        Batch size is equal to self.batch_input_length so reset it if needed.
        
        :param latent: List of latent representations (or descriptors)
        :type latent: list
        :param smiles_ref: List of given SMILES to back-calculate their NLL
        :type smiles_ref: list
        :return: List of NLL of sampling smiles_ref given their latent representations (or descriptors)
        :rtype: list
        """

        assert (
                len(latent) <= self.batch_input_length
        ), "Input length must be less than or equal to batch_input_length."

        # Scale inputs if model is trained on scaled data
        if self.scaler is not None:
            latent = self.scaler.transform(latent)

        # Apply PCA to input if model is trained accordingly
        if self.pca is not None:
            latent = self.pca.transform(latent)

        # Decode states and reset the LSTM cells with them, to bias the generation towards the desired properties
        states = self.latent_to_states_model.predict(latent)

        for dec_layer in range(self.dec_layers):
            self.multi_sample_model.get_layer(
                "Decoder_LSTM_" + str(dec_layer)
            ).reset_states(states=[states[2 * dec_layer], states[2 * dec_layer + 1]])

        # Index of input char "^"
        startidx = self.smilesvec1._char_to_int[self.smilesvec1.startchar]
        # Vectorize the input char for all SMILES
        samplevec = np.zeros((self.batch_input_length, 1, self.smilesvec1.dims[-1]))
        samplevec[:, 0, startidx] = 1
        # Initialize arrays to store NLLs and flag if a SMILES is finished
        NLL = np.zeros((self.batch_input_length,))
        finished = np.array([False] * self.batch_input_length)

        # Loop and predict next char
        for i in range(1000):
            o = self.multi_sample_model.predict(
                samplevec, batch_size=self.batch_input_length
            ).squeeze()
            samplechars = []

            for smiles in smiles_ref:
                try:
                    samplechars.append(smiles[i])
                except:
                    # This is a finished SMILES, so "i" exceeds dimensions
                    samplechars.append("$")

            sampleidc = np.asarray(
                [self.smilesvec1._char_to_int[char] for char in samplechars]
            )

            for idx, samplechar in enumerate(samplechars):
                if not finished[idx]:
                    if i != len(smiles_ref[idx]) - 1:
                        samplevec = np.zeros(
                            (self.batch_input_length, 1, self.smilesvec1.dims[-1])
                        )
                        # One-Hot Encode the character
                        for count, sampleidx in enumerate(sampleidc):
                            samplevec[count, 0, sampleidx] = 1
                        # Calculate negative log likelihood for the selected character given the sequence so far
                        NLL[idx] -= np.log(o[idx][sampleidc[idx]])
                    else:
                        finished[idx] = True

            # If all SMILES are finished, i.e. the endchar "$" has been generated, stop the generation
            if finished.sum() == len(finished):
                return NLL

    def summary(self):
        """Echo the training configuration for inspection.
        """

        print(
            "\nModel trained with dataset %s that has maxlen=%d and charset=%s for %d epochs."
            % (self.dataset_name, self.maxlen, self.charset, self.epochs)
        )

        print(
            "noise_std: %.6f, lstm_dim: %d, dec_layers: %d, td_dense_dim: %d, batch_size: %d, codelayer_dim: %d, lr: %.6f."
            % (
                self.noise_std,
                self.lstm_dim,
                self.dec_layers,
                self.td_dense_dim,
                self.batch_size,
                self.codelayer_dim,
                self.lr,
            )
        )

    def get_graphs(self):
        """Export the graphs of the model and its submodels to png files.
        Requires "pydot" and "graphviz" to be installed (pip install graphviz && pip install pydot).
        """

        try:
            from keras.utils import plot_model
            from keras.utils.vis_utils import model_to_dot

            # from IPython.display import SVG

            plot_model(self.model, to_file="model.png")
            plot_model(
                self.latent_to_states_model, to_file="latent_to_states_model.png"
            )
            plot_model(self.batch_model, to_file="batch_model.png")
            if self.mol_to_latent_model is not None:
                plot_model(self.mol_to_latent_model, to_file="mol_to_latent_model.png")

            print("Models exported to png files.")

        except:
            print("Check pydot and graphviz installation.")

    @timed
    def save(self, model_name):
        """Save model in a zip file.
   
        :param model_name: Path to save model in
        :type model_name: str
        """

        with tempfile.TemporaryDirectory() as dirpath:

            # Save the Keras models
            if self.mol_to_latent_model is not None:
                self.mol_to_latent_model.save(dirpath + "/mol_to_latent_model.h5")

            self.latent_to_states_model.save(dirpath + "/latent_to_states_model.h5")
            self.batch_model.save(dirpath + "/batch_model.h5")

            # Exclude unpicklable and unwanted attributes
            excl_attr = [
                "_DDC__mode",
                "_DDC__train_gen",
                "_DDC__valid_gen",
                "_DDC__mol_to_latent_model",
                "_DDC__latent_to_states_model",
                "_DDC__batch_model",
                "_DDC__sample_model",
                "_DDC__multi_sample_model",
                "_DDC__model",
            ]

            # Cannot deepcopy self.__dict__ because of Keras' thread lock so this is
            # bypassed by popping and re-inserting the unpicklable attributes
            to_add = {}
            # Remove unpicklable attributes
            for attr in excl_attr:
                to_add[attr] = self.__dict__.pop(attr, None)

            # Pickle metadata, i.e. almost everything but the Keras models and generators
            pickle.dump(self.__dict__, open(dirpath + "/metadata.pickle", "wb"))

            # Zip directory with its contents
            shutil.make_archive(model_name, "zip", dirpath)

            # Finally, re-load the popped elements for the model to be usable
            for attr in excl_attr:
                self.__dict__[attr] = to_add[attr]

            print("Model saved.")
