FROM python:3.9-slim

WORKDIR /code

COPY requirements.txt /

RUN pip install -r /requirements.txt \
	&& rm -rf /root/.cache

COPY ./ ./

ENV ENVIRONMENT_FILE=".env"

EXPOSE 8050

ENTRYPOINT ["gunicorn", "--config", "gunicorn_config.py", "app:server"]