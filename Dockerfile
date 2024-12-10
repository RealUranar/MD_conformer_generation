FROM python:3.10-bookworm

#Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > installRust.sh \
    && chmod 700 installRust.sh \
    && ./installRust.sh -y  \
	&& rm installRust.sh \
	&& . root/.cargo/env

ENV PATH=/root/.cargo/bin:$PATH
# #Install FeAtomic
RUN pip install --upgrade pip \
	&& git clone https://github.com/metatensor/featomic \
	&& cd featomic \
	&& pip install .

RUN cd /usr/local \
	&& wget https://github.com/grimme-lab/xtb/releases/download/v6.7.0/xtb-6.7.0-linux-x86_64.tar.xz \
	&& tar -xf xtb-6.7.0-linux-x86_64.tar.xz \
	&& rm xtb-6.7.0-linux-x86_64.tar.xz

ENV PATH=/usr/local/xtb-dist/bin:$PATH

RUN cd /usr/local/bin \
	&& wget https://github.com/crest-lab/crest/releases/download/latest/crest-gnu-12-ubuntu-latest.tar.xz \
	&& tar -xf crest-gnu-12-ubuntu-latest.tar.xz \
	&& chmod +x crest \
	&& rm crest-gnu-12-ubuntu-latest.tar.xz

RUN pip install mindlessgen

ENV PATH=/usr/local/bin:$PATH

RUN pip install matplotlib rdkit

RUN mkdir /modules

COPY ConfGeneration /modules/ConfGeneration
COPY main.py /modules/

ENTRYPOINT ["python", "/modules/main.py"]