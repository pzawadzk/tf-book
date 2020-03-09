#!/bin/sh
mkdir "${HOME}/tf-hub-cache"

cp ~/.bash_profile ~/.bash_profile.bkup
echo 'export TFHUB_CACHE_DIR="${HOME}/tf-hub-cache"' >> ~/.bash_profile
source "${HOME}/.bash_profile"

echo $TFHUB_CACHE_DIR
