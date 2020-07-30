FROM registry.cn-shanghai.aliyuncs.com/hukefei/aliali:v0

ADD . /workspace
WORKDIR /workspace
RUN rm -r /tcdata && mkdir /tcdata

CMD sh run.sh