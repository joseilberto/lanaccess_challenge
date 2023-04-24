# Fetch the ultralytics image
FROM ultralytics/ultralytics

# Set UTF-8 enconding for Python by default
ENV LANG C.UTF-8
# Disabling PIP version checks
ENV PIP_DISABLE_PIP_VERSION_CHECK 1
# Speed package installations up
ENV PYTHONUNBEFFERED 1

# Specify the build workdir
WORKDIR /build

# Create a virtual env for the python dependencies
ENV VIRTUAL_ENV /virtual_env
RUN python -m venv $VIRTUAL_ENV
ENV PATH "${VIRTUAL_ENV}/bin:${PATH}"

# Copy and install poetry dependencies
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:${PATH}"
COPY pyproject.toml .
COPY poetry.lock .
RUN . $VIRTUAL_ENV/bin/activate \ 
    && poetry install --no-root --without ci --with compile

# Copy dvc folder and config file and then pull the Pretrained model
COPY .dvc .dvc/
COPY .git .git/
COPY ["Pretrained YOLOv8N two classes.dvc", "."]
COPY data/default.json .
RUN dvc remote modify myremote gdrive_user_credentials_file ./default.json
RUN dvc pull -r myremote "Pretrained YOLOv8N two classes.dvc"

# We then remove all saved files
RUN rm ./default.json "Pretrained YOLOv8N two classes.dvc" poetry.lock \ 
    pyproject.toml

# We copy the detecting objects_script 
COPY detect_objects.py .

# Define entrypoint to be the detect_objects.py file
ENTRYPOINT ["python", "detect_objects.py"]