FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel

# Install just runpod
RUN pip install --no-cache-dir runpod

# Copy handler
COPY pronunciation_handler.py /app/handler.py

WORKDIR /app

CMD ["python", "-u", "/app/handler.py"]