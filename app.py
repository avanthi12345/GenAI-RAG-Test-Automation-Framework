from api.api_client import APIClient

client = APIClient()

response = client.get("/")

print(response.status_code)
print(response.text)