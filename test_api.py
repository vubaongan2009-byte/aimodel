import requests

BASE_URL = "http://127.0.0.1:8000"

def test_home():
    """Test endpoint /"""
    response = requests.get(f"{BASE_URL}/")
    print("Home:", response.status_code, response.json())

def test_predict_sms():
    """Test SMS spam detection"""
    payload = {"mode": "sms", "text": "Khuyen mai lon! Nhan ngay 50% giam gia tai sieu thi ABC"}
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print("SMS Test:", response.status_code, response.json())

def test_predict_bullying_normal():
    """Test Bullying detection - normal text"""
    payload = {"mode": "bullying", "text": "Hôm nay trời đẹp quá, mình đi chơi nhé!"}
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print("Bullying Normal Test:", response.status_code, response.json())

def test_predict_bullying_bullying():
    """Test Bullying detection - bullying text"""
    payload = {"mode": "bullying", "text": "Mày thật ngu ngốc, chẳng ai ưa mày đâu"}
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print("Bullying Test:", response.status_code, response.json())

def test_invalid_mode():
    """Test với mode không hợp lệ"""
    payload = {"mode": "xxx", "text": "Test invalid mode"}
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print("Invalid Mode Test:", response.status_code, response.json())

if __name__ == "__main__":
    test_home()
    test_predict_sms()
    test_predict_bullying_normal()
    test_predict_bullying_bullying()
    test_invalid_mode()
