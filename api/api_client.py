import requests
from config.config import BASE_URL
from config.logging_config import logger


class APIClient:
    """
    Reusable API Client for REST APIs
    """

    def __init__(self):
        self.base_url = BASE_URL

    def get(self, endpoint, headers=None, params=None):
        url = f"{self.base_url}{endpoint}"

        try:
            logger.info(f"GET Request: {url}")

            response = requests.get(
                url=url,
                headers=headers,
                params=params,
                timeout=30
            )

            logger.info(f"Status Code: {response.status_code}")

            return response

        except Exception as e:
            logger.error(f"GET Request Failed: {e}")
            raise

    def post(self, endpoint, payload=None, headers=None):
        url = f"{self.base_url}{endpoint}"

        try:
            logger.info(f"POST Request: {url}")
            logger.info(f"Payload: {payload}")

            response = requests.post(
                url=url,
                json=payload,
                headers=headers,
                timeout=30
            )

            logger.info(f"Status Code: {response.status_code}")

            return response

        except Exception as e:
            logger.error(f"POST Request Failed: {e}")
            raise

    def put(self, endpoint, payload=None, headers=None):
        url = f"{self.base_url}{endpoint}"

        try:
            logger.info(f"PUT Request: {url}")

            response = requests.put(
                url=url,
                json=payload,
                headers=headers,
                timeout=30
            )

            logger.info(f"Status Code: {response.status_code}")

            return response

        except Exception as e:
            logger.error(f"PUT Request Failed: {e}")
            raise

    def delete(self, endpoint, headers=None):
        url = f"{self.base_url}{endpoint}"

        try:
            logger.info(f"DELETE Request: {url}")

            response = requests.delete(
                url=url,
                headers=headers,
                timeout=30
            )

            logger.info(f"Status Code: {response.status_code}")

            return response

        except Exception as e:
            logger.error(f"DELETE Request Failed: {e}")
            raise