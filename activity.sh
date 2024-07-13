#!/bin/bash
set -ex

python cgcn_train_fs.py
python cgcn_inference_fs_inductive.py --meta_learn True --shot 5 --multi_instance False
python cgcn_inference_fs_inductive.py --meta_learn False --shot 5 --multi_instance False
python cgcn_c3d_postprocess_fs.py
# python gtad_postprocess_fs.py
