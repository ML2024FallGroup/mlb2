.PHONY: build up down logs shell migrate makemigrations createsuperuser test conda-shell

build:
	docker-compose build

up:
	docker-compose up

down:
	docker-compose down

logs:
	docker-compose logs -f

shell:
	docker-compose exec web conda run -n myenv python manage.py shell

conda-shell:
	docker-compose exec web bash

migrate:
	docker-compose exec web conda run -n myenv python manage.py migrate

makemigrations:
	docker-compose exec web conda run -n myenv python manage.py makemigrations

createsuperuser:
	docker-compose exec web conda run -n myenv python manage.py createsuperuser

test:
	docker-compose exec web conda run -n myenv python manage.py test

alt:
	docker-compose exec web conda run -n myenv python alt.py

clean:
	docker-compose down -v
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete