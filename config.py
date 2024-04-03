import importlib
import importlib.util
if importlib.util.find_spec('comet_ml'):
    from comet_ml import Experiment
from utils import get_host, get_user

import argparse
import torch

parser = argparse.ArgumentParser()
COMET_ML_APP_KEY = '8WFTXXjK1vqdazipFZMEpskKS'
COMET_PROJECT_NAME = 'mie1520'

"""
Most Relevant
"""

debug = False
gpu = -1
use_comet_ml = True if importlib.util.find_spec('comet_ml') and not debug else False
print("use_comet_ml: ", use_comet_ml)
# use_comet_ml = False
parser.add_argument('--use_comet_ml', default=use_comet_ml)

if use_comet_ml:
    parser.add_argument('--comet_api_key', default=COMET_ML_APP_KEY)

""" 
dataset:
 sentiment suffix for twitter means the negative classes of the original dataset are combined and the other classes are combined for sentiment analysis
 presplit suffix means training and test are predetermined in [dataset]_labels.txt
 small suffix means a very small dataset used for debugging
"""
# dataset = 'twitter_asian_prejudice'
# dataset = 'twitter_asian_prejudice_sentiment'
# dataset = 'r8_presplit'
# dataset = 'ag_presplit'
# dataset = 'twitter_asian_prejudice_small'
# dataset = 'babe-v3'
# dataset = '20ng'
# dataset = '20ng_small'
# dataset = '20ng_medium'
# dataset = 'ohsumed_presplit'
dataset = 'mr_presplit' 
parser.add_argument('--dataset', default=dataset)

print("dataset: ", dataset)

if 'twitter_asian_prejudice' in dataset:
    if 'sentiment' in dataset:
        num_labels = 2
    else:
        num_labels = 4
elif 'ag' in dataset:
    num_labels = 4
elif 'r8' in dataset:
    num_labels = 8
elif 'babe' in dataset:
    num_labels = 22
elif '20ng' in dataset:
    num_labels = 20
elif 'ohsumed' in dataset:
    num_labels = 23
elif 'mr' in dataset:
    num_labels = 2

parser.add_argument('--random_seed', default=3)


"""
Model. Pt1
"""

model = "text_gcn"

model_params = {}
parser.add_argument('--use_edge_weights', default=False)
parser.add_argument('--init_type', default='one_hot_init')
if model == 'text_gcn':
    n = '--model'
    # pred_type = 'softmax'
    # pred_type = 'mlp'
    # pred_type = 'sm_ensemble_mlp'
    # pred_type = 'full_ensemble_mlp'
    pred_type = 'embed_ensemble_mlp'
    # pred_type = 'llm_embed_mlp'
    node_embd_type = 'gcn'
    layer_dim_list = [200, num_labels]
    # layer_dim_list = [200, num_labels]
    num_layers = len(layer_dim_list)
    class_weights = True
    dropout = True
    llm = True
    # llm_model = None
    llm_model = "sentence-transformers/msmarco-distilbert-base-tas-b"
    # llm_model = "sentence-transformers/all-MiniLM-L6-v2"
    # llm_model = "google/gemma-2b"
    s = 'TextGNN:pred_type={},node_embd_type={},num_layers={},layer_dim_list={},act={},' \
        'dropout={},class_weights={},llm={},llm_model={}'.format(
        pred_type, node_embd_type, num_layers, "_".join([str(i) for i in layer_dim_list]), 'relu', dropout, class_weights, llm, llm_model
    )
    model_params = {
        'pred_type': pred_type,
        'node_embd':  node_embd_type,
        'layer_dims': layer_dim_list,
        'class_weights': class_weights,
        'dropout': dropout,
        'llm': llm,
        'llm_model': llm_model
    }
    parser.add_argument(n, default=s)
else:
    raise NotImplementedError

print("{}: {}\n".format(model, model_params))

"""
Sampling
"""
word_window_size = 20
parser.add_argument('--word_window_size', default=word_window_size)
validation_window_size = 20

"""
Validation
"""
parser.add_argument("--validation_window_size", default=validation_window_size)
parser.add_argument("--validation_metric", default="accuracy",
                    choices=["f1_weighted", "accuracy", "loss"])

use_best_val_model_for_inference = True
parser.add_argument('--use_best_val_model_for_inference', default=use_best_val_model_for_inference)

"""
Evaluation.
"""
tvt_ratio = [0.8, 0.1, 0.1]
parser.add_argument('--tvt_ratio', default=tvt_ratio)
parser.add_argument('--tvt_list', default=["train", "test", "val"])


"""
Optimization.
"""

lr = 2e-2
parser.add_argument('--lr', type=float, default=lr)


device = str('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu != -1
             else 'cpu')
parser.add_argument('--device', default=device)


num_epochs = 200
num_epochs = 2 if debug else num_epochs
parser.add_argument('--num_epochs', type=int, default=num_epochs)



"""
Other info.
"""
parser.add_argument('--user', default=get_user())

parser.add_argument('--hostname', default=get_host())

FLAGS = parser.parse_args()

COMET_EXPERIMENT = None
if FLAGS.use_comet_ml:
    hyper_params = vars(FLAGS)
    COMET_EXPERIMENT = Experiment(api_key=COMET_ML_APP_KEY, project_name=COMET_PROJECT_NAME)
    COMET_EXPERIMENT.log_parameters(hyper_params)
    COMET_EXPERIMENT.log_parameters(model_params)
    print("Experiment url, ", COMET_EXPERIMENT.url)
    COMET_EXPERIMENT.add_tag(FLAGS.dataset)
    if 'llm' in model_params and model_params['llm']:
        COMET_EXPERIMENT.add_tag("llm")



