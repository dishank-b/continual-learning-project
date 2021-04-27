wget https://www.dropbox.com/s/kp3my3412u5k9rl/Imagenet_resize.tar.gz?dl=1 --output-document=Imagenet_resize.tar.gz
wget https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz?dl=1 --output-document=LSUN_resize.tar.gz
mkdir -p data/
tar -xf Imagenet_resize.tar.gz -C data/
tar -xf LSUN_resize.tar.gz -C data/

echo "Installing packages"
pip3 install -r requirements.txt
git clone https://www.github.com/lebrice/Sequoia.git
cd Sequoia
pip3 install -e .
pip3 install -r requirements.txt
# To fix torchmeta batch not found bug
pip3 install pytorch-lightning --upgrade