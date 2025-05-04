FROM python:3.11-slim

# Install Rust build essentials and curl
# Need build-essential for C dependencies some Rust crates might have
RUN apt-get update && apt-get install -y curl build-essential pkg-config openssl libssl-dev --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Install Rust using rustup
# Set CARGO_HOME and RUSTUP_HOME to keep things organized if needed, though defaults are fine
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain stable --profile minimal

WORKDIR /app

# Create a non-root user and group
RUN groupadd --system app && useradd --system --gid app app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
# Now pip install should find cargo
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