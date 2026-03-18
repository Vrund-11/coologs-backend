import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from prisma import Prisma
from fastapi import FastAPI

# Initialize the global Prisma client instance
# This instance will be imported into router files for database operations
db = Prisma()

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI Lifecycle Context Manager
    Handles the asynchronous database connection on startup and cleanup on shutdown.
    """
    # Action: Connect to the database (NeonDB/PostgreSQL) when the server starts
    await db.connect()
    
    try:
        # The application runs while this block is yielded
        yield
    finally:
        # Action: Disconnect the Prisma client safely when the server shuts down
        if db.is_connected():
            await db.disconnect()

# Example usage for initializing the FastAPI app with the database lifespan:
# app = FastAPI(lifespan=lifespan)
#
# To use in routers:
# from database import db
# 
# @router.get("/partners")
# async def get_partners():
#     return await db.partner.find_many()
