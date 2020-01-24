import os

#conda dependencies
os.system('conda install numpy')
os.system('conda install pillow=6.1')
os.system('conda install scipy')
os.system('conda install pytorch=1.0.0 torchvision=0.2.1 -c pytorch')
os.system('conda install -c pytorch')
os.system('conda install matplotlib')
os.system('conda install scikit-learn')
os.system('conda install configargparse')
os.system('conda install pandas')
#pip dependencies
os.system('python -m pip install --upgrade pip')
os.system('pip install functional')
os.system('pip install textX==1.8.0')