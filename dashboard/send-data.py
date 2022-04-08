import time
from influxdb import InfluxDBClient
from datetime import datetime
#  setup database
client = InfluxDBClient(host='localhost', port=8086)
client = InfluxDBClient(host='192.168.96.1', port=8086, username='grafana', password='281042', database='grafana')
client.create_database('time_test')
client.get_list_database()
client.switch_database('time_test')

# setup Payload

# data = {
#     "measurement": "stocks",
#     "tags": {
#         "ticker": "TSLA"
#     },
#     "time": datetime.now(),
#     "fileds": {
#         "open": 688.37,
#         "close": 667.93
#     }
# }
data = {
    "measurement": "place01",
    "fields": {
        "vehicle_count": 5,
        "speed_estimate": 24.5,
        "accident_estimate": 0
    }
}

data2 = {
    "measurement": "place01",
    "fields": {
        "vehicle_count": 10,
        "speed_estimate": 23.5,
        "accident_estimate": 0
    }
}

data3 = {
    "measurement": "place01",
    "fields": {
        "vehicle_count": 20,
        "speed_estimate": 27.5,
        "accident_estimate": 1
    }
}
json_payload = []
json_payload.append(data)
time.sleep(20)
client.write_points(json_payload)

json_payload = []
json_payload.append(data2)
time.sleep(20)
client.write_points(json_payload)

json_payload = []
json_payload.append(data3)
time.sleep(20)
client.write_points(json_payload)

