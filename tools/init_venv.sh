#!/usr/bin/env bash

cur_dir=$(realpath "$(dirname $0)/..")

cd $cur_dir
if [ -e venv ]; then
  echo "Please remove a previously virtual environment folder ${cur_dir}/venv."
  exit
fi

virtualenv venv -p python3 --prompt="(tf-toolbox) "
. venv/bin/activate
pip install -r requirements.txt
cd external/cocoapi
2to3 . -w
cd PythonAPI
make install

# Install OpenVino Model Optimizer (optional)
mo_requirements_file="${INTEL_CVSDK_DIR}/deployment_tools/model_optimizer/requirements_tf.txt"
if [ -e "${mo_requirements_file}" ]; then
  pip install -r ${mo_requirements_file}
fi
