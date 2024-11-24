# edge
졸업 프로젝트

```
tar cvf - lstm_model_heavy.h5 | split -b 20m - lstm_heavy.tar.
```

```
cat lstm_heavy.tar.* > lstm_heavy.tar
tar xvf lstm_heavy.tar
```

