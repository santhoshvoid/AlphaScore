web: gunicorn --workers 1 --timeout 120 --preload -b 0.0.0.0:$PORT app:app
worker: celery -A celery_app worker --beat --loglevel=info --concurrency=1
