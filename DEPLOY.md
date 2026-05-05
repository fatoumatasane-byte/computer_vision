# Deploy Guide

## PythonAnywhere

1. Upload project folder (`intel_project`) and model files to `artifacts/`.
2. Open a Bash console and create virtualenv:
   - `python3.10 -m venv venv`
   - `source venv/bin/activate`
   - `pip install -r requirements.txt`
3. In Web tab:
   - Source code: `/home/<username>/intel_project`
   - WSGI config: import `application` from `webapp/wsgi.py`
4. Reload web app.

## Fly.io

1. Update `fly.toml` app name to a unique value.
2. Login and deploy:
   - `fly auth login`
   - `fly launch --no-deploy` (optional first time)
   - `fly deploy`
3. Check:
   - `fly status`
   - `fly logs`

## Notes

- Keep `artifacts/` synced with your latest Kaggle trained models.
- Use same preprocessing and class order for stable predictions.
