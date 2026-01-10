# Letta Stack Data Protection

## Safeguards Implemented

1. **External Volume**
   - PostgreSQL data is stored in an external volume `letta_pgdata`
   - Cannot be accidentally deleted with `docker compose down -v`

2. **Automated Backups**
   - Daily backups at midnight
   - Stored in `/opt/letta/backups`
   - Compressed with gzip
   - 7-day retention policy

## Safe Operations

1. **Starting the stack:**
```bash
docker compose up -d
```

2. **Stopping the stack safely:**
```bash
docker compose down  # Do NOT use -v flag
```

3. **Manual backup:**
```bash
./backup.sh
```

4. **Restore from backup:**
```bash
# First, decompress the backup if it's compressed
gunzip /opt/letta/backups/backup_name.sql.gz

# Then restore
docker exec -i letta-postgres-1 psql -U letta < /opt/letta/backups/backup_name.sql
```

## ⚠️ Warning

Never use `docker compose down -v` as it will delete the database volume. If you need to remove volumes, first ensure you have a backup.