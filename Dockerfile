FROM python:3.10-bullseye 
LABEL author=kirill_klimushin

# Creating new user
RUN usermod -m ${ROOT_USER} /bin/bash
RUN useradd -aG ${ROOT_USER} sudo 

# Initializing working directory 
WORKDIR /project/dir/${ROOT_USER}

# Copying source code
COPY ./deployment ./entrypoint.sh
COPY ./src ./src
COPY poetry.lock ./
COPY pyproject.toml ./
COPY ./flake8.ini ./
COPY ./proj_requirements ./proj_requirements
COPY ./rest ./rest
COPY ./experiments ./experiments
COPY ./monitoring ./monitoring

# Creating directory for storing log files 
RUN mkdir logs

# Installing libraries 
RUN apt-get install gcc
RUN pip install --upgrade pip 

# Installing dependencies using poetry packet manager
RUN poetry export --format=requirements.txt \
--output=proj_requirements/prod_requirements.txt --without-hashes

# installing project requirements
RUN pip install -r proj_requirements/prod_requirements.txt --no-cache

# installing additional modules for fastapi 
RUN pip install 'fastapi[all]' --upgrade
RUN chmod +x ./entrypoint.sh

ENTRYPOINT [ "sh", "entrypoint.sh" ]