import requests
import logging
from dotenv import get_key
from ocean_runner import Algorithm

logger = logging.getLogger(__name__)

COPERNICUS_AUTH_SERVER_URL = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'
S3_KEYS_MANAGER_URL        = "https://s3-keys-manager.cloudferro.com/api/user/credentials"


def request_copernicus_token(username: str, password: str) -> str:
    payload = {
        'username': username,
        'password': password,
        'grant_type': 'password',
        'client_id': 'cdse-public'
    }

    response = requests.post(COPERNICUS_AUTH_SERVER_URL, data=payload, timeout=10)

    if response.status_code == 200:
        return response.json().get('access_token')
    else:
        raise Algorithm.Error(f'Error generating auth token. Status code = {response.status_code}')


def get_copernicus_token() -> str:
    username = get_key('/data/transformations/algorithm', 'copernicus_username')
    password = get_key('/data/transformations/algorithm', 'copernicus_password')
    if not username or not password:
        raise Algorithm.Error('Copernicus username or password not found')
    return request_copernicus_token(username, password)


def request_temp_s3_creds(headers: dict):
    credentials_response = requests.post(S3_KEYS_MANAGER_URL, headers=headers)
    
    if credentials_response.status_code == 200:
        s3_credentials = credentials_response.json()
        logger.info("Temporary S3 credentials created successfully.")
        return s3_credentials
    else:
        logger.error(f"Failed to create temporary S3 credentials. Status code: {credentials_response.status_code}")
        logger.error("Product download aborted.")
        raise Algorithm.Error(f"Failed to create temporary S3 credentials. Status code: {credentials_response.status_code}")