docker run -it \
--runtime=nvidia \
--network=host \
--volume=/run/jtop.sock:/run/jtop.sock \
--volume=$HOME/edge:/home/user/edge \
--name chanwook \
2e5630291b89