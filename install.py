import os

#conda dependencies D2L
os.system('conda install numpy')
os.system('conda install pillow=6.1')
os.system('conda install scipy')
os.system('conda install pytorch=1.0.0 torchvision=0.2.1 -c pytorch')
os.system('conda install -c pytorch')
os.system('conda install matplotlib')
os.system('conda install scikit-learn')
os.system('conda install configargparse')
os.system('conda install pandas')

#conda dependencies NTP
os.system('conda install gensim')
os.system('conda install scikit-learn')
os.system('conda install flask')
os.system('conda install flask-socketio')
os.system('conda install tabulate')
os.system('conda install termcolor')
os.system('conda install pytest')
os.system('conda install pytest-runner')
os.system('conda install pytest-pep8')
os.system('conda install pytest-xdist')
os.system('conda install pytest-cov')

os.system('python -m pip install --upgrade pip')

#pip dependencies D2L
os.system('pip install functional')
os.system('pip install textX==1.8.0')

#pip dependencies NTP
os.system('pip install tensorflow==1.15')
os.system('pip install parsimonious')

