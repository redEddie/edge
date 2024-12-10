# edge
졸업 프로젝트


### Docker images
* nano

`docker pull nvcr.io/nvidia/l4t-tensorflow:r32.7.1-tf2.7-py3`

* orin

`docker pull nvcr.io/nvidia/l4t-tensorflow:r35.3.1-tf2.11-py3`



1. Clone this repo
```
git clone https://github.com/redEddie/edge.git
```

2. Unzip the files
```
cat ds.tar.* > ds.tar
tar xvf ds.tar
```
```
cat lstm_heavy.tar.* > lstm_heavy.tar
tar xvf lstm_heavy.tar
```
```
cat lstm_light.tar.* > lstm_light.tar
tar xvf lstm_light.tar
```
```
cat gru_heavy.tar.* > gru_heavy.tar
tar xvf gru_heavy.tar
```
```
cat gru_light.tar.* > gru_light.tar
tar xvf gru_light.tar
```

`.tar` is generated by the code below.
```
tar cvf - lstm_model_heavy.h5 | split -b 20m - lstm_heavy.tar.
```

3. Meet the library dependencies.

4. Run code to download nltk files.
```
import nltk
nltk.download('stopwords')
```

5. Check the code runs.
```
python3 lstm_eval.py --gpu --memory_growth
```

6. Evaluate the models.
```
chmod +x filename.sh
```
```
./run_lstm.sh
```
