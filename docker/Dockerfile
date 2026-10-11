# python base image
FROM python:3.11-slim

# setting env variables so that python does not write pyc files or buffering logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Setup working directory inside virtual container
WORKDIR /workspace

# install necessary system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# copy requirements and install required python dependencies
COPY requirements.txt /workspace/requirements.txt

RUN pip install --no-cache-dir --upgrade pip&& \
    pip install --no-cache-dir -r requirements.txt

# copy project code directories and production artifacts
COPY ./src /workspace/src
COPY ./app /workspace/app
COPY ./artifacts /workspace/artifacts

# expose port 8000 so the internal container service can communicate with the outside 
EXPOSE 8000

# Uvicorn web cluster engine on container boot
# public network inside the container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]