import fastapi
import logging
from fastapi.middleware import cors
import os
from rest import controllers
from monitoring import monitoring

logger = logging.getLogger("startup_logger")
file_handler = logging.FileHandler(filename='logs/file_handler.log')
logger.addHandler(file_handler)

DEBUG_MODE = os.environ.get("DEBUG_MODE", 1)
VERSION = os.environ.get("VERSION", 1)

ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*")
ALLOWED_HEADERS = os.environ.get("ALLOWED_HEADERS", "*")
ALLOWED_METHODS = os.environ.get("ALLOWED_METHODS", "*")

# initializing web application
application = fastapi.FastAPI(
    version=VERSION,
    debug=DEBUG_MODE
)

try:
    # adding api monitoring REST endpoints
    application.add_api_route(
        path='/api/monitoring/resources/',
        endpoint=monitoring.get_system_resource_consumption,
        description='REST Endpoint for web application resource monitoring'
    )

    # adding API Rest endpoints
    application.add_api_route(
        path="/api/predict/person/class/",
        endpoint=controllers.predict_person_category,
        description='REST Endpoint for predicting persons age category'
    )

    # adding CORS Middleware to the application
    application.add_middleware(
        middleware_class=cors.CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_headers=ALLOWED_HEADERS,
        allow_methods=ALLOWED_METHODS
    )

except Exception as err:
    logger.critical(err)
    raise SystemExit("Failed to initialize application instance, check logs")
