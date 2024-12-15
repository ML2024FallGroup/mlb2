# Genre Prediction Backend

## Prerequisites

Before you begin, ensure you have the following installed on your system:

- Docker Engine (version 20.10.0 or higher recommended)
- Docker Compose (version 2.0.0 or higher recommended)
- Make (usually pre-installed on Unix-based systems)

## Getting Started

1. Clone the repository:

```bash
git clone https://github.com/ML2024FallGroup/mlb2
cd mlb2
```

2. Building the Project

To build the Docker containers for the project:

```bash
make build
```

This command will execute docker build with the appropriate configuration from your Dockerfile.

3. Running the Project

```bash
make build
```

This command will execute docker up and start your containers according to your Docker Compose configuration.

## Common Commands

```bash
# Stop the project
make down

# View logs
make logs

# Clean up Docker resources
make clean
```
