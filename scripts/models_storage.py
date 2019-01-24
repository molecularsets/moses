from moses.vae import VAE, VAETrainer, get_parser as vae_parser
from moses.organ import ORGAN, ORGANTrainer, get_parser as organ_parser
from moses.aae import AAE, AAETrainer, get_parser as aae_parser
from moses.char_rnn import CharRNN, CharRNNTrainer, get_parser as char_rnn_parser
from moses.junction_tree import JTNNVAE, JTreeTrainer, get_parser as junction_tree_parser


class ModelsStorage():

    def __init__(self):
        self._models = {}
        self.add_model('aae', AAE, AAETrainer, aae_parser)
        self.add_model('char_rnn', CharRNN, CharRNNTrainer, char_rnn_parser)
        self.add_model('junction_tree', JTNNVAE, JTreeTrainer, junction_tree_parser)
        self.add_model('vae', VAE, VAETrainer, vae_parser)
        self.add_model('organ', ORGAN, ORGANTrainer, organ_parser)

    def add_model(self, name, class_, trainer_, parser_):
        self._models[name] = { 'class' : class_,
                               'trainer' : trainer_,
                               'parser' : parser_ }

    def get_model_names(self):
        return self._models.keys()

    def get_model_trainer(self, name):
        return self._models[name]['trainer']

    def get_model_class(self, name):
        return self._models[name]['class']

    def get_model_train_parser(self, name):
        return self._models[name]['parser']

