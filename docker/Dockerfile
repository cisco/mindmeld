# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Cisco Spark - Cognitive Collaboration Group                       #
#                                                                   #
# Dockerfile for local dev environment to work with Workbench.      #
#                                                                   #
# __author_email__ = "ktick@cisco.com"                              #
#                                                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Pull base image.
FROM ubuntu:16.04

ENV ES_PKG_NAME elasticsearch-5.5.1

# System packages
RUN apt-get update && \
    apt-get -y install software-properties-common curl vim

# Create directories
RUN mkdir /root/.pip /root/.mmworkbench /data

# Install python pip
RUN apt-get -y install python-pip python3-pip python-dev build-essential && \
    apt-get -y install python-software-properties

# Install Java
RUN echo oracle-java8-installer shared/accepted-oracle-license-v1-1 \
    select true | debconf-set-selections && \
    add-apt-repository -y ppa:webupd8team/java && \
    apt-get update && \
    apt-get install -y oracle-java8-installer && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /var/cache/oracle-jdk8-installer

# Install duckling dependency
RUN apt-get -y update && \
    apt-get -y upgrade tzdata

# set JAVA_HOME
ENV JAVA_HOME /usr/lib/jvm/java-8-oracle

# Install Elasticsearch.
RUN \
    cd / && \
    wget https://artifacts.elastic.co/downloads/elasticsearch/$ES_PKG_NAME.tar.gz && \
    tar xvzf $ES_PKG_NAME.tar.gz && \
    rm -f $ES_PKG_NAME.tar.gz && \
    mv /$ES_PKG_NAME /elasticsearch

RUN useradd -ms /bin/bash mmworkbench
RUN mkdir /elasticsearch/log
RUN chown -R mmworkbench:mmworkbench /elasticsearch /data /var/log


# Add Config Files
COPY ./conf/elasticsearch.yml /elasticsearch/config/elasticsearch.yml

COPY ./conf/pip.conf /root/.pip/pip.conf
COPY ./conf/config.cfg /root/.mmworkbench/config

WORKDIR /root

RUN curl -o /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py
RUN python3 /tmp/get-pip.py
# Get home_assistant blueprint
RUN wget https://devcenter.mindmeld.com/bp/home_assistant/app.tar.gz -P /root/projects/
RUN mkdir home_assistant
RUN tar -xvf /root/projects/app.tar.gz -C ./home_assistant

# Expose ports.
#   - 9200: HTTP
#   - 9300: transport
EXPOSE 9200
EXPOSE 9300
EXPOSE 7150

# system as both 2 and 3, make 3 the default
RUN echo alias python=python3  >> /root/.bashrc
RUN echo alias pip=pip3  >> /root/.bashrc

ENTRYPOINT export LC_ALL=C.UTF-8 && \
    export LANG=C.UTF-8 && \
    pip3 install --pre mmworkbench~=4.0.0rc && \
    pip3 install click-log==0.1.8 mmworkbench && \
    su mmworkbench -c "ES_JAVA_OPTS='-Xms1g -Xmx1g' /elasticsearch/bin/elasticsearch -d" && \
    mmworkbench num-parse --start && \
    python3 -m home_assistant build && \
    python3 -m home_assistant run
