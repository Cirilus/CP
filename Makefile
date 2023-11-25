up:
	poetry run uvicorn app:app --reload --port 8000

local-up:
	docker compose up minio redis createbucket db

worker-up:
	poetry run celery -A tasks.worker:celery worker -l info --pool gevent

flower-up:
	poetry run celery -A tasks.worker:celery worker --loglevel=INFO

.PHONY: migrate-revision
migrate-revision:
	poetry run alembic revision --autogenerate -m $(name)

.PHONY: migrate-up
migrate-up:
	poetry run alembic upgrade $(rev)

.PHONY: test
test:
	poetry run pytest

.PHONY: lint
lint:
	poetry run pylint .