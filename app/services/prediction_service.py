
from datetime import datetime, timezone

from bson import ObjectId
from pymongo.errors import PyMongoError

from app.services.database import get_history_collection


def save_predictions(user_id, dataframe, predictions, source="manual"):
    """
    Save predictions and their complete HRV input values.

    Each row is stored as a separate MongoDB document.
    """

    if len(dataframe) != len(predictions):
        return False, "Input rows and prediction counts do not match."

    if not user_id:
        return False, "User authentication is required."

    try:
        collection = get_history_collection()
        created_at = datetime.now(timezone.utc)

        documents = []

        for (_, row), prediction in zip(
            dataframe.iterrows(), predictions
        ):
            documents.append({
                "user_id": str(user_id),
                "created_at": created_at,
                "prediction": str(prediction),
                "features": {
                    str(feature): float(value)
                    for feature, value in row.items()
                },
                "source": source
            })

        if not documents:
            return False, "There are no predictions to save."

        collection.insert_many(documents)

        return True, f"{len(documents)} prediction(s) saved successfully."

    except (PyMongoError, TypeError, ValueError) as error:
        return False, f"Unable to save predictions: {error}"


def get_user_predictions(user_id, limit=100):
    """Retrieve prediction history for one authenticated user."""

    try:
        collection = get_history_collection()

        records = list(
            collection.find(
                {"user_id": str(user_id)}
            ).sort("created_at", -1).limit(limit)
        )

        return records, None

    except PyMongoError as error:
        return [], f"Unable to retrieve history: {error}"


def delete_user_prediction(user_id, record_id):
    """Delete a record only if it belongs to the authenticated user."""

    try:
        collection = get_history_collection()

        result = collection.delete_one({
            "_id": ObjectId(record_id),
            "user_id": str(user_id)
        })

        if result.deleted_count == 1:
            return True, "Prediction record deleted successfully."

        return False, "Record not found or you do not have permission to delete it."

    except (PyMongoError, ValueError) as error:
        return False, f"Unable to delete prediction: {error}"


def get_user_prediction_count(user_id):
    """Return the complete prediction count for one user."""

    try:
        collection = get_history_collection()

        return collection.count_documents({
            "user_id": str(user_id)
        }), None

    except PyMongoError as error:
        return None, f"Unable to count predictions: {error}"
