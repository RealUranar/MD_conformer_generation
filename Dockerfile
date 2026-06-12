FROM anaconda/miniconda:latest

RUN apt-get update \
    && apt-get install -y curl build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

#Install Rust and featomic
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > installRust.sh \
    && chmod 700 installRust.sh \
    && ./installRust.sh -y  \
	&& rm installRust.sh \
	&& . root/.cargo/env \
    && pip install --upgrade pip \
    && pip install git+https://github.com/metatensor/featomic


#Install Crest
RUN cd /usr/local/bin \
	&& wget https://github.com/crest-lab/crest/releases/download/latest/crest-gnu-12-ubuntu-latest.tar.xz \
	&& tar -xf crest-gnu-12-ubuntu-latest.tar.xz && mv crest crest_ && mv crest_/crest . \
	&& chmod +x crest \
	&& rm -rf crest-gnu-12-ubuntu-latest.tar.xz crest_

ENV PATH=/root/.cargo/bin:$PATH
ENV PATH=/usr/local/bin:$PATH

ENV OMP_STACKSIZE=5G
ENV OMP_NUM_THREADS=12,1

#Install xtb
RUN conda config --add channels conda-forge \
    && conda install xtb \
    && conda clean -a 

COPY . /app

RUN pip install --no-cache-dir /app \
    && rm -rf /app

RUN pip install \
    mindlessgen matplotlib rdkit ase 
