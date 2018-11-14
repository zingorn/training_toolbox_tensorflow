#!/usr/bin/env bash

if [ -d ${OPEN_VINO_DIR}/bin ]; then
  echo "OpenVino already installed."
  exit 0
fi

echo "Download archives..."
wget -O $HOME/openvino.tgz http://registrationcenter-download.intel.com/akdlm/irc_nas/${OPENVINO_DOWNLOAD_ID}/l_openvino_toolkit_p_${OPENVINO_VERSION}.tgz
tar -xf $HOME/openvino.tgz -C $HOME

echo "Installing the OpenVino dependencies..."
cd $HOME/l_openvino_toolkit_p_${OPENVINO_VERSION}
./install_cv_sdk_dependencies.sh
sed -i 's/ACCEPT_EULA=.*/ACCEPT_EULA=accept/' silent.cfg

echo "Installing the OpenVino toolkit..."
sh ./install.sh -s silent.cfg

