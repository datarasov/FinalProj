from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 500
    assert response.json() == {"message": "DEMO application for the course AI Apps"}
    
def test_predict_positive():
    response = client.post("/predict/",json={"text": "Я люблю рыбалку"})
    json_data = response.json()
    assert response.status_code == 200
    assert json_data['label'] == 'POSITIVE'

def test_predict_negative():
    response = client.post("/predict/",json={"text": "Я не люблю рыбалку"})
    json_data = response.json()
    assert response.status_code == 200
    assert json_data['label'] == 'NEGATIVE'
