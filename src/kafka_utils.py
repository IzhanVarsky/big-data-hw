import kafka

from ansible_credential_utils import read_credentials_from_file

CKPT_TOPIC = "kafka-ckpt"
PREDICTIONS_TOPIC = "kafka-predictions"
ANSIBLE_KAFKA_CREDENTIALS_FILEPATH = 'kafka.credentials'


def get_kafka_credentials_from_vault(ansible_password):
    data = read_credentials_from_file(ANSIBLE_KAFKA_CREDENTIALS_FILEPATH, ansible_password)
    return data.split()


def get_producer(kafka_host, kafka_port) -> kafka.KafkaProducer:
    return kafka.KafkaProducer(bootstrap_servers=f"{kafka_host}:{kafka_port}")


def get_consumer(kafka_host, kafka_port, topic, value_deserializer) -> kafka.KafkaConsumer:
    return kafka.KafkaConsumer(
        topic,
        bootstrap_servers=f"{kafka_host}:{kafka_port}",
        value_deserializer=value_deserializer,
        fetch_max_wait_ms=300_000,  # 5 minutes
        auto_offset_reset="earliest"
    )
