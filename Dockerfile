FROM python:3.10-bookworm

#Install Rascaline
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > installRust.sh \
    && chmod 700 installRust.sh \
    && ./installRust.sh -y \
    && source $HOME/.cargo/env \
    && pip install 'rascaline @ git+https://github.com/Luthaf/rascaline.git'

RUN cd /usr/local \
	&& wget https://github.com/grimme-lab/xtb/releases/download/v6.7.0/xtb-6.7.0-linux-x86_64.tar.xz \
	&& tar -xf xtb-6.7.0-linux-x86_64.tar.xz \
	&& rm xtb-6.7.0-linux-x86_64.tar.xz

ENV PATH=/usr/local/xtb-dist/bin:$PATH

RUN cd /usr/local/bin \
	&& wget https://github.com/crest-lab/crest/releases/download/latest/crest-latest.tar.xz \
	&& tar -xf crest-latest.tar.xz \
	&& chmod +x crest \
	&& rm crest-latest.tar.xz

ENV PATH=/usr/local/bin:$PATH

RUN pip install matplotlib rdkit

RUN mkdir /modules

COPY ConfGeneration /modules/ConfGeneration
COPY startup.py /modules/

ENTRYPOINT ["python", "/modules/startup.py"]