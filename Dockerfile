FROM continuumio/miniconda3:26.3.2

RUN apt-get update \
    && apt-get install -y curl build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

#Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > installRust.sh \
    && chmod 700 installRust.sh \
    && ./installRust.sh -y  \
	&& rm installRust.sh \
	&& . root/.cargo/env

ENV PATH=/root/.cargo/bin:$PATH
ENV PATH=/usr/local/bin:$PATH

ENV OMP_STACKSIZE=5G
ENV OMP_NUM_THREADS=12,1

RUN conda config --add channels conda-forge \
    && conda install xtb \
    && conda clean -a \
    && rm -rf /var/lib/apt/lists/*


RUN pip install \
    mindlessgen matplotlib rdkit ase \
    && rm -rf /var/lib/apt/lists/*