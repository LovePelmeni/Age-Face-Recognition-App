import fastapi 
import logging 
from fastapi.middleware import cors 
import os

logger = logging.getLogger("startup_logger")
file_handler = logging.FileHandler(filename='file_handler')
logger.addHandler(file_handler)

DEBUG_MODE = os.environ.get("DEBUG_MODE", 1)
VERSION = os.environ.get("VERSION", 1)

ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*")
ALLOWED_HEADERS = os.environ.get("ALLOWED_HEADERS", "*")
ALLOWED_METHODS = os.environ.get("ALLOWED_METHODS", "*")

try:
    application = fastapi.FastAPI(
        version=VERSION,
        debug=DEBUG_MODE
    )

    application.add_middleware(
        middleware_class=cors.CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_headers=ALLOWED_HEADERS,
        allow_methods=ALLOWED_METHODS
    )

except Exception as err:
    logger.critical(err)
    raise SystemExit("Failed to initialize application instance, check logs")