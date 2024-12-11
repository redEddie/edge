docker run -it \
--runtime=nvidia \
--network=host \
--volume=/run/jtop.sock:/run/jtop.sock \
--volume=$HOME/edge:/root/edge \
--name chanwook \
8485e8e113fd
