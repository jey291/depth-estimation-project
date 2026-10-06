import requests
import time

url = "http://192.168.4.1/capture"

while True:
    start = time.perf_counter()

    try:
        r = requests.get(
            url,
            timeout=(2, 3)
        )

        r.raise_for_status()

        ms = (
            time.perf_counter()
            - start
        ) * 1000

        print(
            f"{ms:.1f} ms | "
            f"{len(r.content)} bytes"
        )

    except Exception as e:
        print("ERROR:", e)
        time.sleep(1)