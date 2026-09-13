FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONHASHSEED=42 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /runner

COPY requirements-core.txt /runner/requirements-core.txt
RUN pip install --no-cache-dir -r /runner/requirements-core.txt \
    && pip install --no-cache-dir --no-deps aideml==0.2.2

COPY entrypoint.sh /runner/entrypoint.sh
RUN chmod 0555 /runner/entrypoint.sh
ENTRYPOINT ["/runner/entrypoint.sh"]
