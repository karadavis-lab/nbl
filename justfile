set shell := ["fish", "-c"]

# asias r := rye-sync

# sync the rye workspace
rye-sync:
    rye sync --update-all --all-features

# Postgres DB server
start-db:
    docker compose -f db_compose.yaml up -d

stop-db:
    docker compose -f db_compose.yaml stop

down-db:
    docker compose -f db_compose.yaml down
