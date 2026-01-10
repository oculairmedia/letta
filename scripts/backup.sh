#!/bin/bash

# Configuration
BACKUP_DIR="/opt/letta/backups"
RETENTION_DAYS=7
DB_NAME="letta"
DB_USER="letta"
CONTAINER_NAME="letta-postgres-1"

# Ensure backup directory exists
mkdir -p "$BACKUP_DIR"

# Generate timestamp for backup file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/letta_backup_$TIMESTAMP.sql"

# Create backup
echo "Creating backup: $BACKUP_FILE"
docker exec $CONTAINER_NAME pg_dump -U $DB_USER $DB_NAME > "$BACKUP_FILE"

# Check if backup was successful
if [ $? -eq 0 ]; then
    echo "Backup completed successfully"
    
    # Compress the backup
    gzip "$BACKUP_FILE"
    echo "Backup compressed: $BACKUP_FILE.gz"
    
    # Remove old backups
    find "$BACKUP_DIR" -name "letta_backup_*.sql.gz" -mtime +$RETENTION_DAYS -delete
    echo "Old backups cleaned up"
else
    echo "Backup failed!"
    exit 1
fi

# Create external volume if it doesn't exist
if ! docker volume ls | grep -q "letta_pgdata"; then
    echo "Creating external volume: letta_pgdata"
    docker volume create letta_pgdata
fi