''' Defines the functions necessary for calculating the Frechet Inception
Distance (FCD) to evalulate generative models for molecules.

The FCD metric calculates the distance between two distributions of molecules.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by the generative
model.

The FID is calculated by assuming that X_1 and X_2 are the activations of
the preulitmate layer of the CHEMBLNET for generated samples and real world
samples respectivly.
'''

import os
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model
from scipy import linalg


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1:    The mean of the activations of preultimate layer of the
               CHEMNET ( like returned by the function 'get_predictions')
               for generated samples.
    -- mu2:    The mean of the activations of preultimate layer of the
               CHEMNET ( like returned by the function 'get_predictions')
               for real samples.
    -- sigma1: The covariance matrix of the activations of preultimate layer of the
               CHEMNET ( like returned by the function 'get_predictions')
               for generated samples.
    -- sigma2: The covariance matrix of the activations of preultimate layer of the
               CHEMNET ( like returned by the function 'get_predictions')
               for real samples.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def build_masked_loss(loss_function, mask_value):
    """Builds a loss function that masks based on targets

    Args:
        loss_function: The loss function to mask
        mask_value: The value to mask in the targets

    Returns:
        function: a loss function that acts like loss_function with masked inputs
    """

    def masked_loss_function(y_true, y_pred):
        mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
        return loss_function(y_true * mask, y_pred * mask)

    return masked_loss_function


def masked_accuracy(y_true, y_pred):
    mask_value = 0.5
    a = K.sum(K.cast(K.equal(y_true, K.round(y_pred)), K.floatx()))
    c = K.sum(K.cast(K.not_equal(y_true, mask_value), K.floatx()))
    acc = (a) / c
    return acc


def get_one_hot(smiles, pad_len=-1):
    one_hot = ['C', 'N', 'O', 'H', 'F', 'Cl', 'P', 'B', 'Br', 'S', 'I',
               'Si',
               '#', '(', ')', '+', '-', '1', '2', '3', '4', '5', '6',
               '7', '8', '=', '[', ']', '@',
               'c', 'n', 'o', 's', 'X', '.']
    smiles = smiles + '.'
    if pad_len < 0:
        vec = np.zeros((len(smiles), len(one_hot)))
    else:
        vec = np.zeros((pad_len, len(one_hot)))
    cont = True
    j = 0
    i = 0
    while cont:
        if smiles[i + 1] in ['r', 'i', 'l']:
            sym = smiles[i:i + 2]
            i += 2
        else:
            sym = smiles[i]
            i += 1
        if sym in one_hot:
            vec[j, one_hot.index(sym)] = 1
        else:
            vec[j, one_hot.index('X')] = 1
        j += 1
        if smiles[i] == '.' or j >= (pad_len - 1) and pad_len > 0:
            vec[j, one_hot.index('.')] = 1
            cont = False
    return (vec)


def myGenerator_predict(smilesList, batch_size=128, pad_len=350):
    while 1:
        N = len(smilesList)
        nn = pad_len
        idxSamples = np.arange(N)

        for j in range(int(np.ceil(N / batch_size))):
            idx = idxSamples[j * batch_size: min((j + 1) * batch_size, N)]

            x = []
            for i in range(0, len(idx)):
                currentSmiles = smilesList[idx[i]]
                smiEnc = get_one_hot(currentSmiles, pad_len=nn)
                x.append(smiEnc)

            x = np.asarray(x) / 35
            yield x


def load_ref_model(model_file):
    masked_loss_function = build_masked_loss(K.binary_crossentropy, 0.5)
    model = load_model(model_file,
                       custom_objects={
                           'masked_loss_function': masked_loss_function,
                           'masked_accuracy': masked_accuracy
                       })
    model.pop()
    model.pop()
    return model


def get_predictions(smiles, gpu=-1, batch_size=128):
    assert isinstance(gpu, int), "GPU should be an integer"
    model_dir = os.path.split(__file__)[0]
    model_path = os.path.join(model_dir, 'ChemNet_v0.13_pretrained.h5')
    cuda_old = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    order_old = os.environ.get("CUDA_DEVICE_ORDER", None)
    os.environ["CUDA_VISIBLE_DEVICES"] = (str(gpu) if gpu != -1 else "")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    if gpu != -1:
        device = "/gpu:{}".format(gpu)
    else:
        device = "/cpu"
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.device(device):
        sess = tf.Session(config=config)
        set_session(sess)
        K.clear_session()
        model = load_ref_model(model_path)
        smiles_act = model.predict_generator(
             myGenerator_predict(smiles, batch_size=batch_size),
             steps=np.ceil(len(smiles)/batch_size)
        )
        K.clear_session()
        sess.close()
    if cuda_old is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_old)
    else:
        os.environ.pop("CUDA_VISIBLE_DEVICES")
    if order_old is not None:
        os.environ["CUDA_DEVICE_ORDER"] = str(order_old)
    else:
        os.environ.pop("CUDA_DEVICE_ORDER") 
    return smiles_act
