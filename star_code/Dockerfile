FROM python:3.8.10

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy your code and data files
COPY src/ src/


# Create directory for outputs
RUN mkdir -p outputs

CMD ["python", "src/main.py"]
