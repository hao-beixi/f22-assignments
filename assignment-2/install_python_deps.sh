# update pip if need be
pip3 install --user -U pip

# install tensorflow and other python deps
pip3 install -U --user tensorflow==2.10.0 tensorboard==2.10.1 scipy==1.9.2 matplotlib==3.6.1 scikit-learn==1.1.2 gdown==4.5.1 testresources==2.0.1

# install urdf parser for imitation learning 
pip3 install --user -U urdf-parser-py==0.0.4
