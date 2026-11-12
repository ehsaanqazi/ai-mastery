This quiz covers the whole Cloud Computing module: the structure and core services of AWS, Google Cloud and Azure, object storage, virtual machines, GPU sizing, managed databases, model hosting, serverless computing, IAM and security, and monitoring. Aim for at least 13 out of 16 before moving on to Production AI Systems.

```quiz
[
  {
    "q": "Which pairing of provider and its main billing and isolation unit is correct?",
    "options": [
      "AWS: resource group",
      "Google Cloud: project",
      "Azure: availability zone",
      "Google Cloud: subscription"
    ],
    "answer": 1,
    "why": "AWS uses accounts, Google Cloud uses projects (linked to a billing account), and Azure uses subscriptions with resource groups inside them."
  },
  {
    "q": "Which set lists the managed ML platforms of AWS, Google Cloud and Azure, in that order?",
    "options": [
      "Bedrock, BigQuery, Cosmos DB",
      "Lambda, Cloud Run, Azure Functions",
      "SageMaker AI, Vertex AI, Azure Machine Learning",
      "EKS, GKE, AKS"
    ],
    "answer": 2,
    "why": "These are the platforms for training and hosting your own models. Lambda, Cloud Run and Functions are serverless compute; EKS, GKE and AKS are Kubernetes."
  },
  {
    "q": "You set a monthly budget with an alert at 100 percent of forecast spend. What happens when the forecast crosses it?",
    "options": [
      "You receive a notification; resources keep running unless you or automation stop them",
      "All resources are stopped immediately",
      "The account is moved to the free tier",
      "New API calls are rejected until the next month"
    ],
    "answer": 0,
    "why": "Budgets alert. Stopping spend is up to you, which is why maximum replica counts and cleanup habits matter."
  },
  {
    "q": "On Google Cloud, `gcloud storage ls` works but a Python script using `storage.Client()` fails with a credentials error on your laptop. What fixes it?",
    "options": [
      "`gcloud config set project`",
      "Downloading a service account key into the repository",
      "`gcloud services enable storage.googleapis.com`",
      "`gcloud auth application-default login`"
    ],
    "answer": 3,
    "why": "Client libraries use Application Default Credentials, which `gcloud auth application-default login` creates locally."
  },
  {
    "q": "What does this call return?\n\n```python\ns3.generate_presigned_url(\"get_object\", Params={\"Bucket\": \"b\", \"Key\": \"r.pdf\"}, ExpiresIn=600)\n```",
    "options": [
      "The bytes of `r.pdf`",
      "A URL that lets anyone holding it download `r.pdf` for 10 minutes",
      "A URL that makes bucket `b` public for 10 minutes",
      "A lifecycle rule that deletes `r.pdf` after 600 days"
    ],
    "answer": 1,
    "why": "A presigned URL grants one operation on one object until it expires; the bucket stays private."
  },
  {
    "q": "Training checkpoints are written daily and rarely read after a month. Which feature keeps their storage cost under control automatically?",
    "options": [
      "Object versioning",
      "Presigned URLs",
      "Lifecycle rules that transition old checkpoints to colder classes and expire them",
      "Strong read-after-write consistency"
    ],
    "answer": 2,
    "why": "Lifecycle rules move and delete objects by age and prefix without manual work."
  },
  {
    "q": "A training job runs on spot instances. What is the most important design requirement?",
    "options": [
      "Checkpoint regularly to object storage and resume from the latest checkpoint on restart",
      "Use the largest instance type available",
      "Store checkpoints only on the local NVMe disk",
      "Disable the firewall so the job can be restarted remotely"
    ],
    "answer": 0,
    "why": "Spot capacity can be reclaimed at short notice; durable checkpoints make interruptions cheap. Local NVMe is lost when the instance goes."
  },
  {
    "q": "Approximately how much GPU memory do the FP16 weights of a 7-billion-parameter model take?",
    "options": [
      "3.5 GB",
      "7 GB",
      "28 GB",
      "14 GB"
    ],
    "answer": 3,
    "why": "7e9 parameters × 2 bytes = 14e9 bytes, about 14 GB (13.04 GiB). The KV cache and overhead come on top."
  },
  {
    "q": "For an illustrative model with 32 layers, 8 key-value heads, head dimension 128 and FP16 values, what does this compute?\n\n```python\n2 * 32 * 8 * 128 * 2\n```",
    "options": [
      "The size of the weights in bytes",
      "The KV cache size per token in bytes (131,072)",
      "The number of attention heads",
      "The maximum context length"
    ],
    "answer": 1,
    "why": "Keys and values (2) × layers × KV heads × head dimension × bytes per value gives KV bytes per token: 131,072 bytes, or 128 KiB."
  },
  {
    "q": "Your app on 40 autoscaled replicas, each with a pool of 10 connections, overwhelms a managed PostgreSQL instance. What is the best fix?",
    "options": [
      "Disable automated backups to free memory",
      "Increase each replica's pool to 20",
      "Put a connection pooler (RDS Proxy, PgBouncer) in front of the database and keep per-process pools small",
      "Switch to object storage"
    ],
    "answer": 2,
    "why": "A pooler multiplexes many client connections over a small number of real database connections."
  },
  {
    "q": "Which is the most common sensible starting point for vector search in a RAG system that already uses managed PostgreSQL?",
    "options": [
      "The `pgvector` extension in the same database",
      "A separate DynamoDB table per document",
      "Storing embeddings as JSON files in a bucket and scanning them",
      "A dedicated GPU VM running a custom index"
    ],
    "answer": 0,
    "why": "`pgvector` keeps vectors next to metadata and permissions, so filtering is plain SQL and there is no second system to sync."
  },
  {
    "q": "A service receives 20 requests per second with an average latency of 2 seconds. By Little's law, how many requests are in flight on average?",
    "options": [
      "10",
      "22",
      "0.1",
      "40"
    ],
    "answer": 3,
    "why": "In flight = arrival rate × latency = 20 × 2 = 40. Divide by the concurrency one replica supports to size the replica count."
  },
  {
    "q": "Which workload is the worst fit for AWS Lambda?",
    "options": [
      "Chunking a newly uploaded PDF and putting chunks on a queue",
      "Serving a 7B LLM for interactive chat",
      "A nightly job that calls a hosted model API to score 200 support tickets",
      "A Slack bot that forwards questions to a RAG API"
    ],
    "answer": 1,
    "why": "Lambda has no GPUs, a 15-minute limit, a 10 GB image limit and cold starts, all of which rule out hosting a large model."
  },
  {
    "q": "In an AWS policy, why does this statement fail to allow listing the bucket?\n\n```json\n{\"Effect\": \"Allow\", \"Action\": \"s3:ListBucket\", \"Resource\": \"arn:aws:s3:::data/*\"}\n```",
    "options": [
      "`ListBucket` is a bucket-level action, so the resource must be the bucket ARN `arn:aws:s3:::data`",
      "The `Version` field is missing from the statement",
      "`Allow` must be written in lower case",
      "Listing requires `s3:GetObject` instead"
    ],
    "answer": 0,
    "why": "Object ARNs never match bucket-level actions; `GetObject` uses object ARNs, `ListBucket` uses the bucket ARN."
  },
  {
    "q": "What does OIDC workload identity federation give a GitHub Actions deployment pipeline?",
    "options": [
      "A permanent access key stored in GitHub secrets",
      "Administrator access to every cloud account",
      "Short-lived cloud credentials obtained by exchanging a signed job token, with no stored keys",
      "Encryption of the repository at rest"
    ],
    "answer": 2,
    "why": "The cloud trusts tokens from the CI provider under conditions you set (repository, branch) and issues temporary credentials."
  },
  {
    "q": "An alarm fires when p95 latency exceeds 3 seconds for 3 consecutive one-minute periods. Which per-minute series makes it fire at the final minute?",
    "options": [
      "`[3.5, 3.2, 2.8]`",
      "`[2.0, 2.5, 2.9]`",
      "`[4.0, 2.0, 4.0]`",
      "`[2.1, 3.3, 3.1, 3.6]`"
    ],
    "answer": 3,
    "why": "Only the last series has three consecutive values above 3.0 at the end (3.3, 3.1, 3.6)."
  }
]
```
