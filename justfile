set shell := ["fish", "-c"]

# Postgres DB server
start-db:
    docker compose -f db_compose.yaml up -d

stop-db:
    docker compose -f db_compose.yaml stop

down-db:
    docker compose -f db_compose.yaml down
