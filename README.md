# edge
졸업 프로젝트

```
git clone https://github.com/redEddie/edge.git
```

---

```
tar cvf - lstm_model_heavy.h5 | split -b 20m - lstm_heavy.tar.
```

---

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

---

```
import nltk
nltk.download('stopwords')
```

```
python3 lstm_eval.py --gpu --memory_growth
```

---

```
chmod +x filename.sh
```
```
./run_lstm.sh
```
