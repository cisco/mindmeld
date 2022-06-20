# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Cisco Cognitive Collaboration Group                               #
#                                                                   #
# Dockerfile for local dev environment to work with MindMeld.       #
#                                                                   #
# __author_email__ = "mindmeld@cisco.com"                           #
#                                                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Pull base image.
FROM elasticsearch:7.17.4

# System packages
RUN apt-get update && \
    DEBIAN_FRONTEND="noninteractive" apt-get -y install software-properties-common build-essential curl vim tzdata wget

# Create directories
RUN mkdir /root/.pip /root/.mindmeld /data

# Install Python 3.6
RUN add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get install --no-install-recommends -y python3.6 python3.6-dev python3.6-distutils
# Register Python 3.6 as the preferred version to use when running python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1

RUN useradd -ms /bin/bash mindmeld
RUN mkdir -p /elasticsearch/log
RUN chown -R mindmeld:mindmeld /elasticsearch /usr/share/elasticsearch /data /var/log

# Add Config Files
COPY ./conf/elasticsearch.yml /usr/share/elasticsearch/config/elasticsearch.yml

# Add Cluster Status Script
COPY ./wait_for_cluster_to_turn_green.sh /wait_for_cluster_to_turn_green.sh

WORKDIR /root

RUN curl -o /tmp/get-pip.py https://bootstrap.pypa.io/pip/3.6/get-pip.py
RUN python3 /tmp/get-pip.py

RUN pip3 install --upgrade pip
# Explicitly install numpy to support arm builds
RUN pip3 install -U numpy~=1.15
# Install spacy with mindmeld to ensure a version new enough is installed to install en_core_web_sm
RUN pip3 install -U mindmeld spacy
RUN pip3 install click-log==0.1.8

# Add English spacy model or else mindmeld will try to download it itself and fail
RUN python3 -m spacy download en_core_web_sm --default-timeout=1000

ARG REBUILD_BLUEPRINT=unknown
RUN mkdir -p home_assistant && \
    wget https://blueprints.mindmeld.com/home_assistant/app.tar.gz -P /root/projects/ && \
    tar -xvf /root/projects/app.tar.gz -C ./home_assistant

RUN export LC_ALL=C.UTF-8 && \
    export LANG=C.UTF-8 && \
    su mindmeld -c "ES_JAVA_OPTS='-Xms1g -Xmx1g' /usr/share/elasticsearch/bin/elasticsearch -d" && \
    mindmeld num-parse --start && \
    python3 -m home_assistant build

# Expose ports.
#   - 9200: HTTP
#   - 9300: transport
EXPOSE 9200
EXPOSE 9300
EXPOSE 7150

ENTRYPOINT export LC_ALL=C.UTF-8 && \
    export LANG=C.UTF-8 && \
    su mindmeld -c "ES_JAVA_OPTS='-Xms1g -Xmx1g' /usr/share/elasticsearch/bin/elasticsearch -d" && \
    mindmeld num-parse --start && \
    /wait_for_cluster_to_turn_green.sh && \
    python3 -m home_assistant run
