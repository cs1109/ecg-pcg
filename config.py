from numpy.lib.npyio import load
from torch.nn.modules.activation import Threshold


DATASET_DIR="./data"
OUT_DIR="./output"
ftrain = "train_raw.npz"
fval = "val_raw.npz"
ftest = "test_raw.npz"

n_classes = 2
epoch_num = 100
eval_epoch = 5
learning_rate = 1e-3
batch_size = 50
reg_par = 1e-5
decay_rate = 0.7

mod_name = "RTNet"
load_model=False
testing=False
dim_out = 50
dim_query = 50
threshold = 0.24065852165222168

TYPE_LABELS={"正常":0, "异常":1}
LABELS_TYPE={0:"正常", 1:"异常"}
