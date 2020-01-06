FROM nvcr.io/nvidia/pytorch:19.09-py3
COPY requirements.txt /opt/requirements.txt
RUN pip install -r /opt/requirements.txt
