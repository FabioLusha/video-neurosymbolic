FROM python:3.8.10

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install -r requirements.txt

# Create project directories
RUN mkdir -p src outputs data


# run the script in unbuffered mode
CMD ["python", "-u", "src/main.py"]
