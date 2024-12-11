# edge
졸업 프로젝트


### Docker images
* nano

`docker pull nvcr.io/nvidia/l4t-tensorflow:r32.7.1-tf2.7-py3`

* orin

`docker pull nvcr.io/nvidia/l4t-tensorflow:r35.3.1-tf2.11-py3`


# Run

1. Clone this repo at `home` directory
```
git clone https://github.com/redEddie/edge.git
```
2. run `*_run.sh` for container

3. env
```
apt update
apt install tmux nano
```

4. Unzip the files
```
cat ds.tar.* > ds.tar
tar xvf ds.tar
```
```
cat lstm_model.tar.* > lstm_model.tar
tar xvf lstm_model.tar
```
```
cat gru_model.tar.* > gru_model.tar
tar xvf gru_model.tar
```

`.tar` is generated by the code below.
```
tar cvf - lstm_model.h5 | split -b 23m - lstm_model.tar.
```
```
tar cvf - gru_model.h5 | split -b 23m - gru_model.tar.
```

5. Meet the library dependencies.
```
pip3 install --upgrade pip
pip3 install pandas nltk contractions
```
Only for nano
```
pip3 install --upgrade pip 
apt update && apt upgrade -y && apt autoremove -y && apt clean 
python3 -m pip uninstall -y keras 
python3 -m pip install keras==2.7.0
```

7. Run code to download nltk files.
```
import nltk
nltk.download('stopwords')
```

7. Check the code runs.
```
python3 lstm_eval.py --gpu --memory_growth
```

8. Evaluate the models.
```
chmod +x filename.sh
```
```
./run_lstm.sh
```
