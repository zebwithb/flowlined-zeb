FROM python:3.11-slim

WORKDIR /app

# Create a non-root user and group
RUN groupadd --system app && useradd --system --gid app app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the application code
# Make sure your main application entrypoint (src/app.py) is included
COPY ./src ./src
# If you have a main.py at the root that imports from src, copy it too:
# COPY main.py . 

# Change ownership to the non-root user
RUN chown -R app:app /app

# Switch to the non-root user
USER app

EXPOSE 8000

# Run without --reload for production
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]