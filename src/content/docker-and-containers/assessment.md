This quiz covers the whole Docker and Containers module: what containers are, images and the container lifecycle, writing Dockerfiles, Docker Compose, networking and persistent storage. Aim for at least 13 out of 16 before moving on to MLOps.

```quiz
[
  {
    "q": "What is the most accurate description of a container?",
    "options": [
      "A lightweight virtual machine with its own kernel",
      "An isolated process on the host kernel, restricted by namespaces and cgroups",
      "A compressed copy of a virtual environment",
      "A Python package format"
    ],
    "answer": 1,
    "why": "Containers share the host kernel. Namespaces limit what the process can see and cgroups limit what it can use."
  },
  {
    "q": "Which statement about images and containers is correct?",
    "options": [
      "A container is a read-only template; an image is a running instance",
      "Running a container modifies its image",
      "An image is read-only layers plus config; a container adds a writable layer and a process",
      "Each container needs its own full copy of the image on disk"
    ],
    "answer": 2,
    "why": "Images are immutable. Containers share the image's read-only layers and add a thin writable layer."
  },
  {
    "q": "Why should production deployments avoid the `latest` tag?",
    "options": [
      "It is always an unstable development build",
      "Registries reject it",
      "It is a movable pointer, so different machines can run different code and rollback is unclear",
      "It disables layer caching"
    ],
    "answer": 2,
    "why": "Tags can be re-pushed. Deploy a versioned or commit-hash tag, or a digest, to know exactly what is running."
  },
  {
    "q": "What does `docker exec -it api bash` do?",
    "options": [
      "Starts a new container from the `api` image",
      "Opens a shell inside the already running `api` container",
      "Rebuilds the `api` image",
      "Attaches to the container's main process output"
    ],
    "answer": 1,
    "why": "`exec` runs an extra command in a running container; `run` would create a new one."
  },
  {
    "q": "Given this Dockerfile, which steps rebuild after you edit only `app/main.py`?\n\n```dockerfile\nFROM python:3.12-slim\nWORKDIR /srv\nCOPY requirements.txt .\nRUN pip install -r requirements.txt\nCOPY app/ ./app/\n```",
    "options": [
      "All of them",
      "Only `COPY app/ ./app/`",
      "`RUN pip install` and everything after it",
      "None, Docker never rebuilds on code changes"
    ],
    "answer": 1,
    "why": "Earlier steps are unchanged and their inputs are unchanged, so they hit the cache. Only the final copy of the code misses."
  },
  {
    "q": "A Dockerfile has `ENTRYPOINT [\"python\", \"embed.py\"]`. What runs for `docker run embedder --batch-size 64`?",
    "options": [
      "`--batch-size 64` on its own",
      "`python embed.py --batch-size 64`",
      "`python embed.py`, ignoring the arguments",
      "An error, because ENTRYPOINT cannot take arguments"
    ],
    "answer": 1,
    "why": "Arguments to `docker run` are appended to the ENTRYPOINT. With CMD they would replace it instead."
  },
  {
    "q": "Why use the exec form `CMD [\"uvicorn\", ...]` rather than the shell form `CMD uvicorn ...`?",
    "options": [
      "The shell form is not supported by Docker",
      "In the shell form `/bin/sh` is PID 1 and does not forward SIGTERM, so shutdown is not graceful",
      "The exec form makes the image smaller",
      "The exec form runs as a non-root user"
    ],
    "answer": 1,
    "why": "With the exec form your server is PID 1 and receives the stop signal directly."
  },
  {
    "q": "What is a `.dockerignore` file for?",
    "options": [
      "Listing images Docker should not pull",
      "Excluding files such as `.git`, `.venv` and `.env` from the build context",
      "Hiding containers from `docker ps`",
      "Ignoring failed healthchecks"
    ],
    "answer": 1,
    "why": "It keeps the build context small, avoids leaking secrets into images and prevents irrelevant changes from busting the cache."
  },
  {
    "q": "A gated model must be downloaded during `docker build`. What is the safest way to supply the access token?",
    "options": [
      "`ENV HF_TOKEN=...` in the Dockerfile",
      "`COPY` a token file, then `RUN rm` it",
      "A BuildKit secret mount with `docker build --secret`",
      "Hard-code it in the Python download script"
    ],
    "answer": 2,
    "why": "A secret mount is available only during that RUN step and is never written to an image layer. The other options leave the token in the image."
  },
  {
    "q": "For a CPU-only Python ML service, why are Alpine base images often a poor choice?",
    "options": [
      "Alpine cannot run Python",
      "Alpine uses musl, so many scientific wheels must be compiled from source",
      "Alpine images are larger than Debian slim",
      "Alpine does not support non-root users"
    ],
    "answer": 1,
    "why": "Many prebuilt wheels target glibc. On musl, packages may fall back to slow, fragile source builds. Slim Debian or Ubuntu bases avoid this."
  },
  {
    "q": "What must be true for `docker run --gpus all` to work?",
    "options": [
      "The image must contain the NVIDIA kernel driver",
      "The host needs an NVIDIA driver and the NVIDIA Container Toolkit, and the image's CUDA version must be supported by that driver",
      "Only Docker Desktop on macOS is needed",
      "The image must be based on Alpine"
    ],
    "answer": 1,
    "why": "The driver lives on the host because containers share its kernel; the toolkit exposes the GPU to the container."
  },
  {
    "q": "Plain `depends_on: [db]` in Compose guarantees what?",
    "options": [
      "The database is ready to accept connections before the app starts",
      "The database container is started before the app container",
      "The app restarts if the database fails",
      "The database uses a volume"
    ],
    "answer": 1,
    "why": "Only start order. To wait for readiness, add a healthcheck and use `condition: service_healthy`."
  },
  {
    "q": "Inside the `api` container, your code connects to `redis://localhost:6379` and gets connection refused, although Redis runs as a Compose service. What is the fix?",
    "options": [
      "Publish Redis with `ports: [\"6379:6379\"]`",
      "Use `redis://redis:6379`, the service name",
      "Add `EXPOSE 6379` to the API Dockerfile",
      "Switch to the default bridge network"
    ],
    "answer": 1,
    "why": "`localhost` inside a container is that container. Services reach each other by service name on the Compose network."
  },
  {
    "q": "Uvicorn logs `running on http://127.0.0.1:8000` in a container started with `-p 8000:8000`, but the host cannot connect. Why?",
    "options": [
      "Port 8000 is reserved by Docker",
      "The server only listens on the container's loopback; it must listen on `0.0.0.0`",
      "`-p` requires the host port to differ from the container port",
      "EXPOSE is missing"
    ],
    "answer": 1,
    "why": "Published traffic arrives on the container's network interface, not its loopback. Bind to `0.0.0.0` inside the container."
  },
  {
    "q": "Which storage choice fits a Qdrant index in local development on macOS?",
    "options": [
      "The container's writable layer",
      "A tmpfs mount",
      "A named volume at `/qdrant/storage`",
      "A bind mount of your home directory"
    ],
    "answer": 2,
    "why": "Named volumes persist across container replacement and are faster than bind mounts on Docker Desktop. tmpfs and the writable layer are lost."
  },
  {
    "q": "After running `docker compose down -v`, what happens on the next `docker compose up`?",
    "options": [
      "Everything starts with the previous data",
      "Named volumes are recreated empty, so databases start fresh and models download again",
      "Compose refuses to start until volumes are restored",
      "Only bind mounts are reset"
    ],
    "answer": 1,
    "why": "`-v` deletes the project's named volumes along with all data in them."
  }
]
```
