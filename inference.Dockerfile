FROM python:3.7
COPY requirements-drive.txt /tmp/requirements-drive.txt
RUN pip install -r /tmp/requirements-drive.txt