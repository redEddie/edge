# edge
졸업 프로젝트

```
tar cvf - lstm_model_heavy.h5 | split -b 20m - lstm_heavy.tar.
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

