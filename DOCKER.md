# Docker Deployment Guide

## Running The Oracle Container

### Option 1: Using Environment Variables (Recommended)

Pass environment variables directly:

```bash
docker run -p 8000:8000 \
  -e ANTHROPIC_API_KEY=your-key \
  -e TAVILY_API_KEY=your-key \
  -e SUPABASE_URL=your-url \
  -e SUPABASE_KEY=your-key \
  oracle
```

### Option 2: Using .env File

If your `.env` file is properly formatted (simple `KEY=value` pairs, no spaces around `=`), you can use:

```bash
docker run -p 8000:8000 --env-file .env oracle
```

**Important:** The `.env` file must be in this format:
```env
ANTHROPIC_API_KEY=sk-ant-...
TAVILY_API_KEY=tvly-...
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=xxx
```

**Common Issues:**
- No spaces around `=` sign
- No quotes needed (unless the value itself contains spaces)
- One key-value pair per line
- No shell script syntax (no `cat`, `EOF`, etc.)

### Option 3: Using Docker Compose

Create a `docker-compose.yml`:

```yaml
version: '3.8'

services:
  oracle:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - TAVILY_API_KEY=${TAVILY_API_KEY}
      - SUPABASE_URL=${SUPABASE_URL}
      - SUPABASE_KEY=${SUPABASE_KEY}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

Then run:
```bash
docker-compose up
```

## Testing the API

Once the container is running:

```bash
# Health check
curl http://localhost:8000/health

# Run research query
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the AI Application Layer?"}'
```

## Troubleshooting

### .env File Format Issues

If you get errors about `.env` file format:
1. Ensure each line is `KEY=value` (no spaces)
2. Remove any shell script syntax
3. Use individual `-e` flags instead of `--env-file`

### Check Container Logs

```bash
docker logs <container-id>
```

### Interactive Shell

```bash
docker run -it --entrypoint /bin/bash oracle
```
