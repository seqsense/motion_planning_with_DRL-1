FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
# 自動アップグレード
ENV DEBIAN_FRONTEND "noninteractive"
RUN apt-get update -y
RUN apt-get -y \
    -o Dpkg::Options::="--force-confdef" \
        -o Dpkg::Options::="--force-confold" dist-upgrade

# 使いたいソフトウェアを入れる
RUN apt-get install -y --no-install-recommends \
    sudo ssh \
    build-essential \
    zsh screen cmake unzip git curl wget vim tree htop \
    python-dev python-pip python-setuptools \
    python3-dev python3-pip python3-setuptools \
    graphviz python-tk

# キャッシュを消してイメージを小さくする
RUN apt-get clean -y
RUN apt-get autoremove -y
RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get autoremove -y
RUN apt-get autoclean -y
RUN rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# ユーザ設定
# ユーザ ID をパラメータにすることでホストボリュームに対する操作をユーザ権限で実行するようにしている。
# `--userns-remap` の設定が面倒なときに使う。
ARG user_name=amsl
ARG user_id=1942
ARG group_name=amsl
ARG group_id=1942

RUN groupadd -g ${group_id} ${group_name}
RUN useradd -u ${user_id} -g ${group_id} -d /home/${user_name} --create-home --shell /usr/bin/zsh ${user_name}
RUN echo "${user_name} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
RUN chown -R ${user_name}:${group_name} /home/${user_name}
RUN chsh -s /usr/bin/zsh ${user_name}

# 以後 ubuntu ユーザの設定
USER amsl
WORKDIR /home/amsl
ENV HOME /home/amsl

# zsh
#RUN bash -c "$(curl -fsSL https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"
#RUN echo 'shell "/usr/bin/zsh"' >>  /home/ubuntu/.screenrc

ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

WORKDIR $HOME/

RUN pip install tensorflow-gpu
RUN pip install gym matplotlib

#RUN git clone https://github.com/Jumpei-Arima/DeepRL.git
