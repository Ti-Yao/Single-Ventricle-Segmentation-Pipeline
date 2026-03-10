FROM tensorflow/tensorflow:2.9.1

RUN apt-get update && \
    apt-get install -y libopenexr-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install --no-cache-dir \
    importlib-metadata==3.10.1 \
    importlib-resources==5.9.0 \
    json5==0.9.28 \
    jsonref==1.1.0 \
    jsonschema==4.20.0 \
    jsonschema-specifications==2023.11.2 \
    keras==2.9.0 \
    Keras-Preprocessing==1.1.2 \
    matplotlib==3.5.3 \
    matplotlib-inline==0.1.7 \
    neptune==1.8.6 \
    neptune-tensorflow-keras==2.2.2 \
    nibabel==5.2.1 \
    numpy==1.22.4 \
    oauthlib==3.2.0 \
    OpenEXR==1.3.8 \
    openpyxl==3.1.5 \
    pandas==2.0.3 \
    Pillow==9.2.0 \
    plotly==5.10.0 \
    protobuf==3.19.4 \
    pydicom==2.2.2 \
    pylibjpeg==1.4.0 \
    pylibjpeg-libjpeg==1.3.4 \
    pylibjpeg-openjpeg==1.3.2 \
    rasterio==1.3.11 \
    scikit-image==0.20.0 \
    scikit-learn==1.2.2 \
    scipy==1.9.0 \
    seaborn==0.13.0 \
    Shapely==1.8.1.post1 \
    statsmodels==0.14.1 \
    tensorflow==2.9.1 \
    tensorflow-datasets==4.6.0 \
    tensorflow-estimator==2.9.0 \
    tensorflow-graphics==2021.12.3 \
    tensorflow-io==0.26.0 \
    tensorflow-io-gcs-filesystem==0.26.0 \
    tensorflow-metadata==1.9.0 \
    tqdm==4.64.0 \
    volumentations==0.1.8 \
    pycap==1.1.2 \
    ipykernel \
    opencv-python-headless==4.7.0.68 \
    ambra-sdk==3.21.1.0.post1

RUN pip uninstall -y protobuf && \
    pip install --no-binary protobuf protobuf==3.20.1
