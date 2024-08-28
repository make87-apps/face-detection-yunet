import time
from datetime import datetime

from make87_messages.text.PlainText_pb2 import PlainText
from make87 import get_topic, PublisherTopic


def main():
    topic = get_topic(name="my_message")

    while True:
        message = PlainText(body=f"Hello, World! Current date and time is {datetime.now()}.")
        topic.publish(message)
        print(f"Published: {message}")
        time.sleep(1)


if __name__ == "__main__":
    main()
