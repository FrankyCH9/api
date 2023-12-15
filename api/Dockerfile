FROM       python:3
LABEL      maintainer="Sawood Alam <@ibnesayeed>"

ENV        REDIS_URL="redis://localhost:6379"

WORKDIR    /app
COPY       requirements.txt /app/
RUN        pip install -r requirements.txt

COPY       main.py /app/
COPY linkenxtraxtor.py /app/
RUN        chmod a+x main.py

CMD ["python", "main.py"]
