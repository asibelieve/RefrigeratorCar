import requests


def hello_world():
    userid = "200000152";
    thinkAppId = "36b4943878cb4f8eab2c6473f0169917"
    Authorization = "Bearer ff39c0a9-6ba0-4860-a6b2-7d4b3b0de766"
    headers = {"tlinkAppId": thinkAppId, "Authorization": Authorization, "Content-type": "application/json",
               "Cache-Control": "no-cache"}
    url = "http://api.tlink.io/api/device/updateDevice"
    # json = "{\"lng\": \"22.707512\",\"linkType\": \"tcp\",\"timescale\": 300,\"userId\": \"200000152\",\"deviceName\": \"测试1\",\"lat\": \"113.978179\",\"deviceId\":\"200053169\",\"sensorList\":[{\"sensorId\":\"200390033\",\"unit\": \".\",\"sensorName\": \"2号传感器111\",\"sensorType\": 1,\"ordernum\": 1,\"decimal\": 0}]]"
    json = "{\"lng\": \"22.707512\",\"linkType\": \"tcp\",\"timescale\": 300,\"userId\": 200000152,\"deviceName\": \"测试2\",\"lat\": \"113.978179\",\"deviceId\":200053169,\"sensorList\": [{" \
           "\"sensorId\":200390033,\"unit\": \"个\",\"sensorName\":\"测试1\",\"sensorType\": 1,\"ordernum\": 1,\"decimal\": 2 }]}"
    r = requests.post(url, data=json.encode("utf-8"), headers=headers)
    print(r.status_code)
    print(r.text)
    return 'Hello World!'


hello_world()
