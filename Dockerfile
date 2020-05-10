FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

WORKDIR /home/user

# apt-get
RUN apt-get update
RUN apt-get -y full-upgrade
RUN apt-get install -y python2.7 python-pip ffmpeg git vim

# git
RUN git clone https://github.com/imatge-upc/wav2pix.git

# pip
RUN pip install torch==1.4.0 torchvision==0.5.0
RUN pip install numpy scipy future visdom pyyaml

# make pickle
RUN python /home/user/wav2pix/scripts/generate_pickle.py --dataset_path /home/user/wav2pix/dataset/train/ --pickle_path /home/user/wav2pix/pickle/ --save_name train_pickle
RUN python /home/user/wav2pix/scripts/generate_pickle.py --dataset_path /home/user/wav2pix/dataset/test/ --pickle_path /home/user/wav2pix/pickle/ --save_name test_pickle
