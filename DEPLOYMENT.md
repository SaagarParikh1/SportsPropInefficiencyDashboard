# Deployment Guide

This project is a Streamlit application. It should be deployed on a host that can run a persistent Python web process with:

```bash
streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT
```

## Recommended: Render

The repo includes `render.yaml`, so Render can deploy it as a Python Web Service.

1. Create a new Render Web Service from this GitHub repo.
2. Use the `main` branch.
3. Build command:

```bash
pip install -r requirements.txt
```

4. Start command:

```bash
streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT
```

5. Add this environment variable in Render:

```bash
THE_ODDS_API_KEY=your_api_key_here
```

## Also Good: Streamlit Community Cloud

Use this repo, the `main` branch, and `app.py` as the main file.

Add `THE_ODDS_API_KEY` through Streamlit's app secrets/settings so the live prop scanner can run without local settings.

## Why Vercel Does Not Work Well Here

Vercel's Python runtime expects a serverless function or web framework object such as `app`, `application`, or `handler`. Streamlit apps do not expose one of those variables. They are launched by running the Streamlit server process.

That is why Vercel shows an error like:

```text
Found app.py but it does not export a top-level "app", "application", or "handler" variable.
```

For this project, use Render, Streamlit Community Cloud, Railway, Fly.io, or another host that supports long-running Python web processes.
