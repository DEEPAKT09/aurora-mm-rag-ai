
# railway_service.py

import os
import requests
from dotenv import load_dotenv

load_dotenv()

RAILRADAR_BASE_URL = "https://api.railradar.in/v1"


def get_train_information(
    train_number: str,
    info_type: str = "details",
    journey_date: str | None = None,
) -> dict:
    """
    Fetch train information from RailRadar.

    Args:
        train_number: Indian train number, e.g. "12919".
        info_type: "details" or "live".
        journey_date: Optional date in YYYY-MM-DD format.

    Returns:
        Dictionary containing API data or an error.
    """

    api_key = os.getenv("RAILRADAR_API_KEY")

    if not api_key:
        return {
            "success": False,
            "error": "RAILRADAR_API_KEY is missing from the environment."
        }

    train_number = str(train_number).strip()

    if not train_number.isdigit() or len(train_number) != 5:
        return {
            "success": False,
            "error": "Please provide a valid 5-digit train number."
        }

    if info_type not in {"details", "live"}:
        return {
            "success": False,
            "error": "info_type must be 'details' or 'live'."
        }

    if journey_date:
        from datetime import date

        try:
            date.fromisoformat(journey_date)
            if len(journey_date) != 10:
                raise ValueError
        except ValueError:
            return {
                "success": False,
                "error": "Journey date must use YYYY-MM-DD format."
            }

    if info_type == "live":
        endpoint = f"{RAILRADAR_BASE_URL}/trains/{train_number}/live"
        params = {}

        if journey_date:
            params["date"] = journey_date
    else:
        endpoint = (
            f"{RAILRADAR_BASE_URL}/legacy/trains/{train_number}"
        )
        params = {"dataType": "full"}

        if journey_date:
            params["journeyDate"] = journey_date

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }

    try:
        response = requests.get(
            endpoint,
            headers=headers,
            params=params,
            timeout=15,
        )

        if response.status_code == 401:
            return {
                "success": False,
                "error": "RailRadar rejected the API key. Check your credentials."
            }

        if response.status_code == 404:
            return {
                "success": False,
                "error": f"No train information found for {train_number}."
            }

        if response.status_code == 429:
            return {
                "success": False,
                "error": "RailRadar request limit reached. Please try later."
            }

        response.raise_for_status()
        result = response.json()

        if not result.get("success", False):
            return {
                "success": False,
                "error": result.get("error", {}).get(
                    "message", "RailRadar could not retrieve train information."
                )
            }

        return {
            "success": True,
            "info_type": info_type,
            "data": result.get("data", {}),
            "meta": result.get("meta", {}),
        }

    except requests.Timeout:
        return {
            "success": False,
            "error": "RailRadar timed out. Please try again."
        }

    except requests.RequestException as exc:
        return {
            "success": False,
            "error": f"RailRadar request failed: {exc}"
        }

    except ValueError:
        return {
            "success": False,
            "error": "RailRadar returned an invalid JSON response."
        }
