# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Cisco Cognitive Collaboration Group                               #
#                                                                   #
# Dockerfile for local dev environment to work with MindMeld.       #
#                                                                   #
# __author_email__ = "mindmeld@cisco.com"                           #
#                                                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Pull base image.
FROM ubuntu:18.04

ENV ES_PKG_NAME elasticsearch-7.8.0-linux-x86_64
ENV ES_FOLDER_NAME elasticsearch-7.8.0

# System packages
RUN apt-get update && \
    apt-get -y install software-properties-common curl vim

# Create directories
RUN mkdir /root/.pip /root/.mindmeld /data

# Install python pip
RUN apt-get -y install python-pip python3-pip python-dev build-essential && \
    apt-get -y install software-properties-common

# Install Java
RUN apt update
RUN apt-get update
RUN apt-get install -y wget
RUN apt install openjdk-8-jdk -y

# Install duckling dependency
RUN apt-get -y update && \
    DEBIAN_FRONTEND="noninteractive" apt-get -y upgrade tzdata

# set JAVA_HOME
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64

# Install Elasticsearch.
RUN \
    cd / && \
    wget https://artifacts.elastic.co/downloads/elasticsearch/$ES_PKG_NAME.tar.gz && \
    tar xvzf $ES_PKG_NAME.tar.gz && \
    rm -f $ES_PKG_NAME.tar.gz && \
    mv /$ES_FOLDER_NAME /elasticsearch

RUN useradd -ms /bin/bash mindmeld
RUN mkdir /elasticsearch/log
RUN chown -R mindmeld:mindmeld /elasticsearch /data /var/log


# Add Config Files
COPY ./conf/elasticsearch.yml /elasticsearch/config/elasticsearch.yml

WORKDIR /root

RUN curl -o /tmp/get-pip.py https://bootstrap.pypa.io/pip/3.6/get-pip.py
RUN python3 /tmp/get-pip.py

# system as both 2 and 3, make 3 the default
RUN echo alias python=python3  >> /root/.bashrc
RUN echo alias pip=pip3  >> /root/.bashrc

RUN pip3 install --upgrade pip
RUN pip3 install -U mindmeld
RUN pip3 install click-log==0.1.8

RUN export LC_ALL=C.UTF-8 && \
    export LANG=C.UTF-8 && \
    mindmeld num-parse --start

# Expose ports.
#   - 9200: HTTP
#   - 9300: transport
EXPOSE 9200
EXPOSE 9300
EXPOSE 7151

ENTRYPOINT export LC_ALL=C.UTF-8 && \
    export LANG=C.UTF-8 && \
    pip3 install mindmeld --upgrade && \
    su mindmeld -c "ES_JAVA_OPTS='-Xms1g -Xmx1g' /elasticsearch/bin/elasticsearch"

