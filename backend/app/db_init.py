import logging
import asyncio
from sqlalchemy import text
from app.database import engine, Base
# Import models so they are registered in metadata
from app.models import Document, Folder, Log

logger = logging.getLogger(__name__)

async def init_db():
    """
    Initialize database: wait for connection and create tables.
    """
    logger.info("Initializing database...")
    
    # Wait for DB connection
    retries = 0
    while retries < 60:
        try:
            async with engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            logger.info("Database connection established.")
            break
        except Exception as e:
            retries += 1
            logger.warning(f"Waiting for database ({retries}/60)... Error: {e}")
            await asyncio.sleep(1)
            
    if retries >= 60:
        raise Exception("Could not connect to database after 60 seconds.")

    # Create tables
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully.")
            
            # Here we can add initial data or specific migrations if needed
            # For Postgres, standard create_all handles schemas.
            
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise e
