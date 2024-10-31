from locust import HttpUser, task, between

class QuickstartUser(HttpUser):

    host = "http://localhost:8000"
    wait_time = between(1, 2)

    @task
    def test_chat(self):
        self.client.post("/chat", json={"text": "你好，请介绍一下自己"})

    @task
    def test_batch_chat(self):
        self.client.post("localhost:8000/batch_chat", json={"queries": ["问题1", "问题2"]})