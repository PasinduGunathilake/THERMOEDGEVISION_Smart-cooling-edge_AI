import os
from pymongo import MongoClient
import certifi
from dotenv import load_dotenv
from Smart_cooling_edge_AI.logging.logger import get_logger


# logger initialization
logger = get_logger("DatabaseLogger")


load_dotenv()
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
CONNECTION_URI = os.getenv("MONGODB_URI")

client = None
data_base = None
collection = None

# check env variables are loded or not
if not CONNECTION_URI:
    logger.warning("WARNING: MONGODB_URI is missing from environment variables (.env file missing, empty, or incorrect). Database logging will be disabled.")
elif not DB_NAME or not COLLECTION_NAME:
    logger.warning("WARNING: DB_NAME or COLLECTION_NAME is missing from environment variables. Database logging will be disabled.")
else:
    # try connect to db
    try:
        logger.info("Attempting to connect to MongoDB Atlas...")
        client = MongoClient(CONNECTION_URI, tlsCAFile=certifi.where())
        client.admin.command('ping')
        logger.info("Pinged your deployment. Successfully connected to MongoDB!")

        data_base = client[DB_NAME]
        collection = data_base[COLLECTION_NAME]

    except Exception as e:
        logger.critical(f"MongoDB Connection Error: {str(e)}", exc_info=True)
        client = None
        data_base = None
        collection = None

    


def db_system_logging(person_count, hvac_data, image_path, timestamp_str):
    """
    Pushes clean structured metadata to MongoDB Atlas for analytics.
    Fails gracefully without crashing the main AI pipeline if DB is offline.
    """
    if collection is None:
        # show warning 
        logger.warning("Database collection is offline or not configured. Skipping database logging for this frame.")
        return

    log_document = {
        "timestamp": timestamp_str,
        "student_count": person_count,
        "occupancy_level": hvac_data["occupancy_level"],
        "hvac_control": {
            "ac_active_count": hvac_data["ac_active_count"],
            "ac_mode": hvac_data["ac_mode"],
            "target_temp_min": hvac_data["temp_min"],
            "target_temp_max": hvac_data["temp_max"],
            "fan_speed": hvac_data["fan_speed"],
            "louver_position": hvac_data["louver_position"]
        },
        "image_path": image_path
    }
    
    try:
        collection.insert_one(log_document)
        logger.info("Structured data successfully logged to MongoDB Atlas.")
    except Exception as e:
        logger.error(f"Failed to insert data into MongoDB: {str(e)}", exc_info=True)