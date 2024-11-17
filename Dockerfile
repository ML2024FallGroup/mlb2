# Dockerfile
FROM continuumio/miniconda3

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    make \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy environment file
COPY environment.yml .

# Create conda environment and activate it for subsequent commands
RUN conda env create -f environment.yml && \
    echo "conda activate myenv" >> ~/.bashrc

# Make RUN commands use the conda environment
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# Install additional packages that might not be in conda
RUN pip install python-ffmpeg django-environ spleeter

# Copy project files
COPY . .

# Run as non-root user
RUN useradd -m myuser && chown -R myuser:myuser /app
USER myuser

# Override default shell to ensure conda env is activated
SHELL ["/bin/bash", "-c"]
ENV PATH /opt/conda/envs/myenv/bin:$PATH

# Command to run
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]