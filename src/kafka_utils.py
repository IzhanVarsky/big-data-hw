import kafka
from kafka import KafkaAdminClient
from kafka.admin import NewTopic

from ansible_credential_utils import read_credentials_from_file

CKPT_TOPIC = "kafka-ckpt"
PREDICTIONS_TOPIC = "kafka-predictions"
ANSIBLE_KAFKA_CREDENTIALS_FILEPATH = 'kafka.credentials'


def get_bootstrap_servers(kafka_host, kafka_port):
    return f'{kafka_host}:{kafka_port}'


def create_topics(kafka_host, kafka_port):
    bootstrap_servers = get_bootstrap_servers(kafka_host, kafka_port)
    admin_client = KafkaAdminClient(
        bootstrap_servers=bootstrap_servers,
        client_id='test'
    )

    topic_list = []
    for name in [CKPT_TOPIC, PREDICTIONS_TOPIC]:
        topic_list.append(NewTopic(name=name, num_partitions=1, replication_factor=1))
    admin_client.create_topics(new_topics=topic_list, validate_only=False)


def get_kafka_credentials_from_vault(ansible_password):
    data = read_credentials_from_file(ANSIBLE_KAFKA_CREDENTIALS_FILEPATH, ansible_password)
    host, port = data.split()
    return host, "29092"


def get_producer(kafka_host, kafka_port) -> kafka.KafkaProducer:
    bootstrap_servers = get_bootstrap_servers(kafka_host, kafka_port)
    return kafka.KafkaProducer(bootstrap_servers=bootstrap_servers)


def get_consumer(kafka_host, kafka_port, topic) -> kafka.KafkaConsumer:
    bootstrap_servers = get_bootstrap_servers(kafka_host, kafka_port)
    return kafka.KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        value_deserializer=lambda x: x.decode('utf-8'),
        fetch_max_wait_ms=300_000,  # 5 minutes
        auto_offset_reset="earliest"
    )
