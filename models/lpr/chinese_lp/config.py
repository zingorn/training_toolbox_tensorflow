import os

input_shape = (24, 94, 3)  # (height, width, channels)
use_h_concat = False
use_oi_concat = False

model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'models')  # Path to the folder where all training and evaluation artifacts will be located
if not os.path.exists(model_dir):
  os.makedirs(model_dir)


current_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.normpath(os.path.join(current_dir, "../../.."))

class train:

  train_list_file_path = os.path.join(root_dir, "./data/chinese_lp/train")
  val_list_file_path = os.path.join(root_dir, "./data/chinese_lp/test")

  batch_size = 32
  val_batch_size = 128
  steps = 250000
  learning_rate = 0.001
  grad_noise_scale = 0.001
  opt_type = 'Adam'

  model = 'model'
  start_iter = 0
  snap_iter = 10000
  display_iter = 10
  val_iter = 100
  val_steps = 0  # 0 to run on full dataset

  rnn_cells_num = 128

  apply_basic_aug = False
  apply_stn_aug = True
  apply_blur_aug = False
  stn_alignment = True
  use_lbp=False

  need_to_save_weights = True
  need_to_save_log = True

  class execution:
    CUDA_VISIBLE_DEVICES = "0"  # Environment variable to control CUDA device used for training
    per_process_gpu_memory_fraction = 0.8  # Fix extra memory allocation issue
    allow_growth = True  # Option which attempts to allocate only as much GPU memory based on runtime allocations


class eval:

  model = 'model'
  file_list_path = os.path.join(root_dir, "./data/chinese_lp/test")
  checkpoint = ''
  max_lp_length = 20
  beam_search_width = 10
  rnn_cells_num = 128

  class execution:
    CUDA_VISIBLE_DEVICES = "0"  # Environment variable to control CUDA device used for training
    per_process_gpu_memory_fraction = 0.8  # Fix extra memory allocation issue
    allow_growth = True  # Option which attempts to allocate only as much GPU memory based on runtime allocations



class infer:

  file_list_path = os.path.join(root_dir, "./data/chinese_lp/test_infer")
  checkpoint = '/home/atrushkov/work/workspace/tf_ssd_toolbox_quick_start/tf_ssd_toolbox_test2/tf_ssd_toolbox/samples/lpr/model/model_1/snapshot220000.ckpt'
  max_lp_length = 20
  beam_search_width = 10
  rnn_cells_num = 128

  class execution:
    CUDA_VISIBLE_DEVICES = "0"  # Environment variable to control CUDA device used for training
    per_process_gpu_memory_fraction = 0.8  # Fix extra memory allocation issue
    allow_growth = True  # Option which attempts to allocate only as much GPU memory based on runtime allocations
