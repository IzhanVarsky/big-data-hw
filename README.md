### Классификация изображений датасета FashionMNIST

#### Ход работы

1. Я скачал датасет FashionMNIST и реализовал [простую нейросеть](src/model.py) для классификации изображений
2. Написал [тесты](src/unit_tests) на проверку корректности считывания и разбиения датасета. Также написаны тесты для
   проверки точности и F-меры на тестовой выборке.
3. Гиперпараметры и другая нужная информация записана в файле [config.yaml](config.yaml).
4. Для автоматизации тестирования и развертывания модели были задействованы следующие подходы: Docker, DVC и CI/CD.
5. DVC позволяет скачивать различные данные с удаленного сервера. Вместо того чтобы загружать на гитхаб
   крупные файлы (что почти всегда невозможно), можно загрузить "легкие" конфигурационные файлы DVC, в которых будет
   указано, где в действительности лежат настоящие "тяжелые" файлы. Такой подход позволяет автоматизировать процесс
   получения данных. В моем случае DVC был задействован для датасета и весов предобученной модели.
6. Настройка DVC производилась следующими командами:

* Установка DVC: `pip install dvc[all]`
* `dvc init`
* `dvc add .\data\FashionMNIST\raw` - начать отслеживать данные в папке
* `dvc add .\checkpoints`  - начать отслеживать данные в папке
* `dvc remote add -d storage ssh://izhanvarsky@109.188.135.85/storage/izhanvarsky/big_data_hw1_dvc` - сконфигурировать
  удаленный сервер
* `dvc remote modify storage ask_password true` - настройка авторизации на сервер с помощью пароляы
* `dvc push` - отправить данные на удаленный сервер
* Чтобы получить данные с удаленного сервера, нужно ввести команду `dvc pull`
* Подробности использования DVC можно узнать здесь: https://dvc.org/doc/start/data-management/data-versioning

7.

* [Docker](https://www.docker.com/) позволяет запускать и отлаживать процессы в отдельном контейнере (чаще всего
  представляющим собой
  определенную ОС с различными установленными зависимостями и библиотеками).
* Для создания такой оболочки (окружения)
  используется [Dockerfile](Dockerfile), в котором описывается, какую базовую оболочку использовать и какие зависимости
  в неё необходимо дополнительно добавить.
* Это окружение нужно сбилдить и затем собранный образ можно загрузить на [dockerhub](https://hub.docker.com/) -
  открытое хранилище Docker образов.
* Для запуска контейнера используется [docker-compose.yml](docker-compose.yml), в котором указывается, какие команды
  необходимо выполнить.

8. Автоматизация загрузки датасета и весов, сборки Docker образа и тестирование модели (всё вместе - CI/CD) производится
   с помощью встроенных инструментов GitHub - [GitHub Actions](https://docs.github.com/en/actions).

9.

* CI. Этапы загрузки датасета и весов, а также сборка Docker образа и отправка его на DockerHub описаны в
  файле [build-docker-image.yml](.github/workflows/build-docker-image.yml)
* Подробнее про настройку Docker можно узнать [здесь](https://github.com/docker/build-push-action)
* CD. Получение образа с DockerHub и последующее тестирование модели описано в
  файле [docker_run_tests.yaml](.github/workflows/docker_run_tests.yaml)

10. **HW2**: Реализация общения с БД Greenplum

* В [docker-compose.yml](docker-compose.yml) поднимается Greenplum контейнер, который инициализирует свои таблицы,
  используя [init.sql](init.sql).
* Используя библиотеку `greenplumpython` можно читать/писать данные из/в БД
* Для удобства CI и CD этапы были объединены в один файл [ci_cd.yml](.github%2Fworkflows%2Fci_cd.yml)
* В GutHub Actions были добавлены credentials для доступа к БД.
* Во время CD этапа эти данные передаются с помощью ENV Variables поднимаемым с
  помощью [docker-compose.yml](docker-compose.yml) контейнерам.

Собственно, сама работа (чтение/запись) с БД:

* Изначально в БД записан путь к чекпоинту, который нужно протестировать.
* В [test_training_results.py](src%2Funit_tests%2Ftest_training_results.py) перед этапом тестирования
  происходит считывание этих данных из БД.
* После тестирования полученные результаты записываются в БД.

11. **HW3**: Использование Ansible Vault для отдельного хранения данных для доступа к БД

* Используя `ansible-vault` (работает только на Linux), я сгенерировал зашифрованный
  файл [db.credentials](db.credentials), в котором хранятся
  данные для доступа к БД (credentials).
* Чтение этих данных написано в файле [ansible_credential_utils.py](src%2Fansible_credential_utils.py)
  (используется питоновская библиотека `ansible-vault`)
* Теперь в [docker-compose.yml](docker-compose.yml) контейнеру-классификатору не нужно передавать напрямую DB
  credentials, нужен только пароль от Ansible Vault.
* Пароль от Ansible Vault хранится в GitHub Actions. Во время CD этапа он передается docker-compose.
* Во время тестирования в [test_training_results.py](src%2Funit_tests%2Ftest_training_results.py) данные для доступа к
  БД теперь не считываются напрямую (было реализовано через ENV variables), а вначале
  считывается пароль от Ansible Vault и затем с его помощью происходит считывание [db.credentials](db.credentials)
  файла с этими данными.

12. **HW4**: Использование Apache Kafka как промежуточного звена между сервисом БД и тестирующем сервисом

* При получении информации о чекпоинте от базы данных тестирующий сервис становится Consumer'ом, а БД - Producer'ом
* Затем при записи результатов тестирования в БД тестирующий сервис становится Producer'ом, а БД - Consumer'ом
* Эти сообщения можно передавать, используя Kafka. В первом случае топиком сообщений будет получение чекпоинта, а во
  втором случае топиком сообщений будет получение метрик
* В [docker-compose.yml](docker-compose.yml) дополнительно создаются контейнеры ZooKeeper и Kafka
* В Ansible Vault ([kafka.credentials](kafka.credentials)) также сохраняются хост и порт, по которым будет происходит
  обращение к Kafka.
* Используя библиотеку `kafka-python`, в [test_training_results.py](src%2Funit_tests%2Ftest_training_results.py) перед
  началом тестирования создаются топики CKPT и METRICS.
* Далее создаются Producer и два вида Consumer'ов. И после этого используется логика, описанная ранее.

#### Дополнительная информация

* Пароли для авторизации на DockerHub и на удаленном сервере были сохранены с использованием GitHub Secrets
* Ссылка на загруженный Docker
  образ: https://hub.docker.com/repository/docker/izhanvarsky/bigdata-hw1-fashion-mnist-classifier
* Результаты тестирования можно найти
  здесь: https://github.com/IzhanVarsky/big-data-hw1/actions/workflows/docker_run_tests.yaml
* Пример результатов тестирования:

```
Creating network "big-data-hw_default" with the default driver
Creating zookeeper ... 
Creating database  ... 
Creating database  ... done
Creating zookeeper ... done
Creating kafka     ... 
Creating kafka     ... done
Creating fashion-mnist-classifier ... 
Creating fashion-mnist-classifier ... done
Attaching to database, zookeeper, kafka, fashion-mnist-classifier
database                    | The files belonging to this database system will be owned by user "***".
database                    | This user must also own the server process.
database                    | 
database                    | The database cluster will be initialized with locale "en_US.utf8".
database                    | The default database encoding has accordingly been set to "UTF8".
database                    | The default text search configuration will be set to "english".
database                    | 
database                    | Data page checksums are disabled.
database                    | 
database                    | fixing permissions on existing directory /var/lib/***ql/data ... ok
database                    | creating subdirectories ... ok
database                    | selecting dynamic shared memory implementation ... posix
database                    | selecting default max_connections ... 100
database                    | selecting default shared_buffers ... 128MB
database                    | selecting default time zone ... Etc/UTC
kafka                       | ===> User
kafka                       | uid=1000(appuser) gid=1000(appuser) groups=1000(appuser)
kafka                       | ===> Configuring ...
kafka                       | ===> Running preflight checks ... 
kafka                       | ===> Check if /var/lib/kafka/data is writable ...
zookeeper                   | ===> User
zookeeper                   | uid=1000(appuser) gid=1000(appuser) groups=1000(appuser)
zookeeper                   | ===> Configuring ...
zookeeper                   | ===> Running preflight checks ... 
zookeeper                   | ===> Check if /var/lib/zookeeper/data is writable ...
zookeeper                   | ===> Check if /var/lib/zookeeper/log is writable ...
zookeeper                   | ===> Launching ... 
database                    | creating configuration files ... ok
zookeeper                   | ===> Launching zookeeper ... 
database                    | running bootstrap script ... ok
database                    | performing post-bootstrap initialization ... ok
database                    | syncing data to disk ... ok
database                    | 
database                    | 
database                    | Success. You can now start the database server using:
database                    | 
database                    |     pg_ctl -D /var/lib/***ql/data -l logfile start
database                    | 
database                    | initdb: warning: enabling "trust" authentication for local connections
database                    | initdb: hint: You can change this by editing pg_hba.conf or using the option -A, or --auth-local and --auth-host, the next time you run initdb.
database                    | waiting for server to start....2023-10-01 14:41:14.322 UTC [48] LOG:  starting PostgreSQL 16.0 (Debian 16.0-1.pgdg120+1) on x86_64-pc-linux-gnu, compiled by gcc (Debian 12.2.0-14) 12.2.0, 64-bit
database                    | 2023-10-01 14:41:14.332 UTC [48] LOG:  listening on Unix socket "/var/run/***ql/.s.PGSQL.5432"
database                    | 2023-10-01 14:41:14.338 UTC [51] LOG:  database system was shut down at 2023-10-01 14:41:14 UTC
database                    | 2023-10-01 14:41:14.354 UTC [48] LOG:  database system is ready to accept connections
database                    |  done
database                    | server started
database                    | 
database                    | /usr/local/bin/docker-entrypoint.sh: running /docker-entrypoint-initdb.d/init.sql
database                    | CREATE TABLE
database                    | INSERT 0 1
database                    | CREATE TABLE
database                    | 
database                    | 
database                    | 2023-10-01 14:41:14.623 UTC [48] LOG:  received fast shutdown request
database                    | waiting for server to shut down....2023-10-01 14:41:14.624 UTC [48] LOG:  aborting any active transactions
database                    | 2023-10-01 14:41:14.646 UTC [48] LOG:  background worker "logical replication launcher" (PID 54) exited with exit code 1
database                    | 2023-10-01 14:41:14.649 UTC [49] LOG:  shutting down
database                    | 2023-10-01 14:41:14.650 UTC [49] LOG:  checkpoint starting: shutdown immediate
database                    | 2023-10-01 14:41:14.670 UTC [49] LOG:  checkpoint complete: wrote 65 buffers (0.4%); 0 WAL file(s) added, 0 removed, 0 recycled; write=0.015 s, sync=0.003 s, total=0.022 s; sync files=48, longest=0.002 s, average=0.001 s; distance=207 kB, estimate=207 kB; lsn=0/151E718, redo lsn=0/151E718
database                    | 2023-10-01 14:41:14.674 UTC [48] LOG:  database system is shut down
database                    |  done
database                    | server stopped
database                    | 
database                    | PostgreSQL init process complete; ready for start up.
database                    | 
database                    | 2023-10-01 14:41:14.776 UTC [1] LOG:  starting PostgreSQL 16.0 (Debian 16.0-1.pgdg120+1) on x86_64-pc-linux-gnu, compiled by gcc (Debian 12.2.0-14) 12.2.0, 64-bit
database                    | 2023-10-01 14:41:14.782 UTC [1] LOG:  listening on IPv4 address "0.0.0.0", port 5432
database                    | 2023-10-01 14:41:14.783 UTC [1] LOG:  listening on IPv6 address "::", port 5432
database                    | 2023-10-01 14:41:14.786 UTC [1] LOG:  listening on Unix socket "/var/run/***ql/.s.PGSQL.5432"
database                    | 2023-10-01 14:41:14.789 UTC [64] LOG:  database system was shut down at 2023-10-01 14:41:14 UTC
database                    | 2023-10-01 14:41:14.806 UTC [1] LOG:  database system is ready to accept connections
kafka                       | ===> Check if Zookeeper is healthy ...
zookeeper                   | [2023-10-01 14:41:19,256] INFO Reading configuration from: /etc/kafka/zookeeper.properties (org.apache.zookeeper.server.quorum.QuorumPeerConfig)
zookeeper                   | [2023-10-01 14:41:19,271] INFO clientPortAddress is 0.0.0.0:2181 (org.apache.zookeeper.server.quorum.QuorumPeerConfig)
zookeeper                   | [2023-10-01 14:41:19,271] INFO secureClientPort is not set (org.apache.zookeeper.server.quorum.QuorumPeerConfig)
zookeeper                   | [2023-10-01 14:41:19,271] INFO observerMasterPort is not set (org.apache.zookeeper.server.quorum.QuorumPeerConfig)
zookeeper                   | [2023-10-01 14:41:19,272] INFO metricsProvider.className is org.apache.zookeeper.metrics.impl.DefaultMetricsProvider (org.apache.zookeeper.server.quorum.QuorumPeerConfig)
zookeeper                   | [2023-10-01 14:41:19,277] INFO autopurge.snapRetainCount set to 3 (org.apache.zookeeper.server.DatadirCleanupManager)
zookeeper                   | [2023-10-01 14:41:19,277] INFO autopurge.purgeInterval set to 0 (org.apache.zookeeper.server.DatadirCleanupManager)
zookeeper                   | [2023-10-01 14:41:19,277] INFO Purge task is not scheduled. (org.apache.zookeeper.server.DatadirCleanupManager)
zookeeper                   | [2023-10-01 14:41:19,278] WARN Either no config or no quorum defined in config, running in standalone mode (org.apache.zookeeper.server.quorum.QuorumPeerMain)
zookeeper                   | [2023-10-01 14:41:19,279] INFO Log4j 1.2 jmx support not found; jmx disabled. (org.apache.zookeeper.jmx.ManagedUtil)
zookeeper                   | [2023-10-01 14:41:19,280] INFO Reading configuration from: /etc/kafka/zookeeper.properties (org.apache.zookeeper.server.quorum.QuorumPeerConfig)
zookeeper                   | [2023-10-01 14:41:19,281] INFO clientPortAddress is 0.0.0.0:2181 (org.apache.zookeeper.server.quorum.QuorumPeerConfig)
zookeeper                   | [2023-10-01 14:41:19,281] INFO secureClientPort is not set (org.apache.zookeeper.server.quorum.QuorumPeerConfig)
zookeeper                   | [2023-10-01 14:41:19,281] INFO observerMasterPort is not set (org.apache.zookeeper.server.quorum.QuorumPeerConfig)
zookeeper                   | [2023-10-01 14:41:19,282] INFO metricsProvider.className is org.apache.zookeeper.metrics.impl.DefaultMetricsProvider (org.apache.zookeeper.server.quorum.QuorumPeerConfig)
zookeeper                   | [2023-10-01 14:41:19,282] INFO Starting server (org.apache.zookeeper.server.ZooKeeperServerMain)
zookeeper                   | [2023-10-01 14:41:19,293] INFO ServerMetrics initialized with provider org.apache.zookeeper.metrics.impl.DefaultMetricsProvider@1fb700ee (org.apache.zookeeper.server.ServerMetrics)
zookeeper                   | [2023-10-01 14:41:19,296] INFO zookeeper.snapshot.trust.empty : false (org.apache.zookeeper.server.persistence.FileTxnSnapLog)
zookeeper                   | [2023-10-01 14:41:19,311] INFO  (org.apache.zookeeper.server.ZooKeeperServer)
zookeeper                   | [2023-10-01 14:41:19,311] INFO   ______                  _                                           (org.apache.zookeeper.server.ZooKeeperServer)
zookeeper                   | [2023-10-01 14:41:19,312] INFO  |___  /                 | |                                          (org.apache.zookeeper.server.ZooKeeperServer)
zookeeper                   | [2023-10-01 14:41:19,312] INFO     / /    ___     ___   | | __   ___    ___   _ __     ___   _ __    (org.apache.zookeeper.server.ZooKeeperServer)
zookeeper                   | [2023-10-01 14:41:19,312] INFO    / /    / _ \   / _ \  | |/ /  / _ \  / _ \ | '_ \   / _ \ | '__| (org.apache.zookeeper.server.ZooKeeperServer)
zookeeper                   | [2023-10-01 14:41:19,312] INFO   / /__  | (_) | | (_) | |   <  |  __/ |  __/ | |_) | |  __/ | |     (org.apache.zookeeper.server.ZooKeeperServer)
zookeeper                   | [2023-10-01 14:41:19,312] INFO  /_____|  \___/   \___/  |_|\_\  \___|  \___| | .__/   \___| |_| (org.apache.zookeeper.server.ZooKeeperServer)
zookeeper                   | [2023-10-01 14:41:19,315] INFO                                               | |                      (org.apache.zookeeper.server.ZooKeeperServer)
zookeeper                   | [2023-10-01 14:41:19,315] INFO                                               |_|                      (org.apache.zookeeper.server.ZooKeeperServer)
zookeeper                   | [2023-10-01 14:41:19,315] INFO  (org.apache.zookeeper.server.ZooKeeperServer)
zookeeper                   | [2023-10-01 14:41:19,317] INFO Server environment:zookeeper.version=3.6.4--d65253dcf68e9097c6e95a126463fd5fdeb4521c, built on 12/18/2022 18:10 GMT (org.apache.zookeeper.server.ZooKeeperServer)
zookeeper                   | [2023-10-01 14:41:19,317] INFO Server environment:host.name=c065d8fa6e13 (org.apache.zookeeper.server.ZooKeeperServer)
zookeeper                   | [2023-10-01 14:41:19,317] INFO Server environment:java.version=11.0.20 (org.apache.zookeeper.server.ZooKeeperServer)
zookeeper                   | [2023-10-01 14:41:19,317] INFO Server environment:java.vendor=Azul Systems, Inc. (org.apache.zookeeper.server.ZooKeeperServer)
zookeeper                   | [2023-10-01 14:41:19,317] INFO Server environment:java.home=/usr/lib/jvm/java-11-zulu-openjdk-ca (org.apache.zookeeper.server.ZooKeeperServer)
zookeeper                   | [2023-10-01 14:41:19,320] INFO Server environment:java.class.path=/usr/bin/../share/java/kafka/netty-codec-4.1.96.Final.jar:/usr/bin/../share/java/kafka/jackson-databind-2.13.4.2.jar:/usr/bin/../share/java/kafka/javassist-3.27.0-GA.jar:/usr/bin/../share/java/kafka/connect-mirror-7.3.5-ccs.jar:/usr/bin/../share/java/kafka/jakarta.annotation-api-1.3.5.jar:/usr/bin/../share/java/kafka/netty-resolver-4.1.96.Final.jar:/usr/bin/../share/java/kafka/scala-reflect-2.13.10.jar:/usr/bin/../share/java/kafka/metrics-core-4.1.12.1.jar:/usr/bin/../share/java/kafka/jetty-util-9.4.51.v20230217.jar:/usr/bin/../share/java/kafka/plexus-utils-3.3.0.jar:/usr/bin/../share/java/kafka/activation-1.1.1.jar:/usr/bin/../share/java/kafka/netty-handler-4.1.96.Final.jar:/usr/bin/../share/java/kafka/scala-library-2.13.10.jar:/usr/bin/../share/java/kafka/javax.servlet-api-3.1.0.jar:/usr/bin/../share/java/kafka/rocksdbjni-7.1.2.jar:/usr/bin/../share/java/kafka/kafka-streams-examples-7.3.5-ccs.jar:/usr/bin/../share/java/kafka/jackson-core-2.13.4.jar:/usr/bin/../share/java/kafka/zookeeper-3.6.4.jar:/usr/bin/../share/java/kafka/jackson-datatype-jdk8-2.13.4.jar:/usr/bin/../share/java/kafka/commons-lang3-3.8.1.jar:/usr/bin/../share/java/kafka/kafka-storage-api-7.3.5-ccs.jar:/usr/bin/../share/java/kafka/netty-common-4.1.96.Final.jar:/usr/bin/../share/java/kafka/argparse4j-0.7.0.jar:/usr/bin/../share/java/kafka/scala-collection-compat_2.13-2.6.0.jar:/usr/bin/../share/java/kafka/netty-transport-classes-epoll-4.1.96.Final.jar:/usr/bin/../share/java/kafka/netty-buffer-4.1.96.Final.jar:/usr/bin/../share/java/kafka/jose4j-0.9.3.jar:/usr/bin/../share/java/kafka/javax.ws.rs-api-2.1.1.jar:/usr/bin/../share/java/kafka/jersey-server-2.34.jar:/usr/bin/../share/java/kafka/kafka-shell-7.3.5-ccs.jar:/usr/bin/../share/java/kafka/connect-runtime-7.3.5-ccs.jar:/usr/bin/../share/java/kafka/connect-mirror-client-7.3.5-ccs.jar:/usr/bin/../share/java/kafka/jackson-module-jaxb-annotations-2.13.4.jar:/usr/bin/../share/java/kafka/jackson-jaxrs-base-2.13.4.jar:/usr/bin/../share/java/kafka/netty-transport-native-epoll-4.1.96.Final.jar:/usr/bin/../share/java/kafka/jetty-io-9.4.51.v20230217.jar:/usr/bin/../share/java/kafka/hk2-api-2.6.1.jar:/usr/bin/../share/java/kafka/swagger-annotations-2.2.0.jar:/usr/bin/../share/java/kafka/jakarta.activation-api-1.2.2.jar:/usr/bin/../share/java/kafka/jopt-simple-5.0.4.jar:/usr/bin/../share/java/kafka/aopalliance-repackaged-2.6.1.jar:/usr/bin/../share/java/kafka/scala-logging_2.13-3.9.4.jar:/usr/bin/../share/java/kafka/jakarta.ws.rs-api-2.1.6.jar:/usr/bin/../share/java/kafka/connect-transforms-7.3.5-ccs.jar:/usr/bin/../share/java/kafka/kafka-server-common-7.3.5-ccs.jar:/usr/bin/../share/java/kafka/lz4-java-1.8.0.jar:/usr/bin/../share/java/kafka/jackson-jaxrs-json-provider-2.13.4.jar:/usr/bin/../share/java/kafka/kafka-streams-test-utils-7.3.5-ccs.jar:/usr/bin/../share/java/kafka/slf4j-api-1.7.36.jar:/usr/bin/../share/java/kafka/connect-json-7.3.5-ccs.jar:/usr/bin/../share/java/kafka/paranamer-2.8.jar:/usr/bin/../share/java/kafka/kafka_2.13-7.3.5-ccs.jar:/usr/bin/../share/java/kafka/metrics-core-2.2.0.jar:/usr/bin/../share/java/kafka/jersey-common-2.34.jar:/usr/bin/../share/java/kafka/maven-artifact-3.8.4.jar:/usr/bin/../share/java/kafka/jakarta.xml.bind-api-2.3.3.jar:/usr/bin/../share/java/kafka/jersey-container-servlet-core-2.34.jar:/usr/bin/../share/java/kafka/jackson-module-scala_2.13-2.13.4.jar:/usr/bin/../share/java/kafka/reflections-0.9.12.jar:/usr/bin/../share/java/kafka/kafka-streams-scala_2.13-7.3.5-ccs.jar:/usr/bin/../share/java/kafka/kafka-raft-7.3.5-ccs.jar:/usr/bin/../share/java/kafka/hk2-locator-2.6.1.jar:/usr/bin/../share/java/kafka/audience-annotations-0.13.0.jar:/usr/bin/../share/java/kafka/kafka-storage-7.3.5-ccs.jar:/usr/bin/../share/java/kafka/jetty-security-9.4.51.v20230217.jar:/usr/bin/../share/java/kafka/netty-transport-native-unix-common-4.1.96.Final.jar:/usr/bin/../share/java/kafka/jetty-continuation-9.4.51.v20230217.jar:/usr/bin/../share/java/kafka/zookeeper-jute-3.6.4.jar:/usr/bin/../share/java/kafka/jetty-http-9.4.51.v20230217.jar:/usr/bin/../share/java/kafka/jersey-container-servlet-2.34.jar:/usr/bin/../share/java/kafka/jetty-server-9.4.51.v20230217.jar:/usr/bin/../share/java/kafka/reload4j-1.2.19.jar:/usr/bin/../share/java/kafka/jackson-dataformat-csv-2.13.4.jar:/usr/bin/../share/java/kafka/zstd-jni-1.5.2-1.jar:/usr/bin/../share/java/kafka/jetty-servlets-9.4.51.v20230217.jar:/usr/bin/../share/java/kafka/jetty-servlet-9.4.51.v20230217.jar:/usr/bin/../share/java/kafka/kafka-metadata-7.3.5-ccs.jar:/usr/bin/../share/java/kafka/jersey-hk2-2.34.jar:/usr/bin/../share/java/kafka/snappy-java-1.1.10.1.jar:/usr/bin/../share/java/kafka/hk2-utils-2.6.1.jar:/usr/bin/../share/java/kafka/commons-lang3-3.12.0.jar:/usr/bin/../share/java/kafka/jersey-client-2.34.jar:/usr/bin/../share/java/kafka/jetty-util-ajax-9.4.51.v20230217.jar:/usr/bin/../share/java/kafka/kafka-tools-7.3.5-ccs.jar:/usr/bin/../share/java/kafka/jline-3.21.0.jar:/usr/bin/../share/java/kafka/jakarta.inject-2.6.1.jar:/usr/bin/../share/java/kafka/jaxb-api-2.3.0.jar:/usr/bin/../share/java/kafka/jackson-annotations-2.13.4.jar:/usr/bin/../share/java/kafka/jakarta.validation-api-2.0.2.jar:/usr/bin/../share/java/kafka/trogdor-7.3.5-ccs.jar:/usr/bin/../share/java/kafka/kafka-clients-7.3.5-ccs.jar:/usr/bin/../share/java/kafka/kafka-log4j-appender-7.3.5-ccs.jar:/usr/bin/../share/java/kafka/netty-transport-4.1.96.Final.jar:/usr/bin/../share/java/kafka/kafka.jar:/usr/bin/../share/java/kafka/scala-java8-compat_2.13-1.0.2.jar:/usr/bin/../share/java/kafka/connect-api-7.3.5-ccs.jar:/usr/bin/../share/java/kafka/kafka-streams-7.3.5-ccs.jar:/usr/bin/../share/java/kafka/osgi-resource-locator-1.0.3.jar:/usr/bin/../share/java/kafka/jetty-client-9.4.51.v20230217.jar:/usr/bin/../share/java/kafka/commons-cli-1.4.jar:/usr/bin/../share/java/kafka/slf4j-reload4j-1.7.36.jar:/usr/bin/../share/java/kafka/connect-basic-auth-extension-7.3.5-ccs.jar:/usr/bin/../share/java/confluent-telemetry/* (org.apache.zookeeper.server.ZooKeeperServer)
zookeeper                   | [2023-10-01 14:41:19,321] INFO Server environment:java.library.path=/usr/java/packages/lib:/usr/lib64:/lib64:/lib:/usr/lib (org.apache.zookeeper.server.ZooKeeperServer)
zookeeper                   | [2023-10-01 14:41:19,321] INFO Server environment:java.io.tmpdir=/tmp (org.apache.zookeeper.server.ZooKeeperServer)
zookeeper                   | [2023-10-01 14:41:19,321] INFO Server environment:java.compiler=<NA> (org.apache.zookeeper.server.ZooKeeperServer)
zookeeper                   | [2023-10-01 14:41:19,321] INFO Server environment:os.name=Linux (org.apache.zookeeper.server.ZooKeeperServer)
zookeeper                   | [2023-10-01 14:41:19,321] INFO Server environment:os.arch=amd64 (org.apache.zookeeper.server.ZooKeeperServer)
zookeeper                   | [2023-10-01 14:41:19,321] INFO Server environment:os.version=6.2.0-1012-azure (org.apache.zookeeper.server.ZooKeeperServer)
zookeeper                   | [2023-10-01 14:41:19,321] INFO Server environment:user.name=appuser (org.apache.zookeeper.server.ZooKeeperServer)
zookeeper                   | [2023-10-01 14:41:19,321] INFO Server environment:user.home=/home/appuser (org.apache.zookeeper.server.ZooKeeperServer)
zookeeper                   | [2023-10-01 14:41:19,321] INFO Server environment:user.dir=/home/appuser (org.apache.zookeeper.server.ZooKeeperServer)
zookeeper                   | [2023-10-01 14:41:19,321] INFO Server environment:os.memory.free=494MB (org.apache.zookeeper.server.ZooKeeperServer)
zookeeper                   | [2023-10-01 14:41:19,321] INFO Server environment:os.memory.max=512MB (org.apache.zookeeper.server.ZooKeeperServer)
zookeeper                   | [2023-10-01 14:41:19,321] INFO Server environment:os.memory.total=512MB (org.apache.zookeeper.server.ZooKeeperServer)
zookeeper                   | [2023-10-01 14:41:19,321] INFO zookeeper.enableEagerACLCheck = false (org.apache.zookeeper.server.ZooKeeperServer)
zookeeper                   | [2023-10-01 14:41:19,321] INFO zookeeper.digest.enabled = true (org.apache.zookeeper.server.ZooKeeperServer)
zookeeper                   | [2023-10-01 14:41:19,321] INFO zookeeper.closeSessionTxn.enabled = true (org.apache.zookeeper.server.ZooKeeperServer)
zookeeper                   | [2023-10-01 14:41:19,321] INFO zookeeper.flushDelay=0 (org.apache.zookeeper.server.ZooKeeperServer)
zookeeper                   | [2023-10-01 14:41:19,321] INFO zookeeper.maxWriteQueuePollTime=0 (org.apache.zookeeper.server.ZooKeeperServer)
zookeeper                   | [2023-10-01 14:41:19,321] INFO zookeeper.maxBatchSize=1000 (org.apache.zookeeper.server.ZooKeeperServer)
zookeeper                   | [2023-10-01 14:41:19,321] INFO zookeeper.intBufferStartingSizeBytes = 1024 (org.apache.zookeeper.server.ZooKeeperServer)
zookeeper                   | [2023-10-01 14:41:19,324] INFO Weighed connection throttling is disabled (org.apache.zookeeper.server.BlueThrottle)
zookeeper                   | [2023-10-01 14:41:19,325] INFO minSessionTimeout set to 6000 (org.apache.zookeeper.server.ZooKeeperServer)
zookeeper                   | [2023-10-01 14:41:19,325] INFO maxSessionTimeout set to 60000 (org.apache.zookeeper.server.ZooKeeperServer)
zookeeper                   | [2023-10-01 14:41:19,326] INFO Response cache size is initialized with value 400. (org.apache.zookeeper.server.ResponseCache)
zookeeper                   | [2023-10-01 14:41:19,326] INFO Response cache size is initialized with value 400. (org.apache.zookeeper.server.ResponseCache)
zookeeper                   | [2023-10-01 14:41:19,327] INFO zookeeper.pathStats.slotCapacity = 60 (org.apache.zookeeper.server.util.RequestPathMetricsCollector)
zookeeper                   | [2023-10-01 14:41:19,327] INFO zookeeper.pathStats.slotDuration = 15 (org.apache.zookeeper.server.util.RequestPathMetricsCollector)
zookeeper                   | [2023-10-01 14:41:19,327] INFO zookeeper.pathStats.maxDepth = 6 (org.apache.zookeeper.server.util.RequestPathMetricsCollector)
zookeeper                   | [2023-10-01 14:41:19,327] INFO zookeeper.pathStats.initialDelay = 5 (org.apache.zookeeper.server.util.RequestPathMetricsCollector)
zookeeper                   | [2023-10-01 14:41:19,327] INFO zookeeper.pathStats.delay = 5 (org.apache.zookeeper.server.util.RequestPathMetricsCollector)
zookeeper                   | [2023-10-01 14:41:19,327] INFO zookeeper.pathStats.enabled = false (org.apache.zookeeper.server.util.RequestPathMetricsCollector)
zookeeper                   | [2023-10-01 14:41:19,330] INFO The max bytes for all large requests are set to 104857600 (org.apache.zookeeper.server.ZooKeeperServer)
zookeeper                   | [2023-10-01 14:41:19,330] INFO The large request threshold is set to -1 (org.apache.zookeeper.server.ZooKeeperServer)
zookeeper                   | [2023-10-01 14:41:19,330] INFO Created server with tickTime 3000 minSessionTimeout 6000 maxSessionTimeout 60000 clientPortListenBacklog -1 datadir /var/lib/zookeeper/log/version-2 snapdir /var/lib/zookeeper/data/version-2 (org.apache.zookeeper.server.ZooKeeperServer)
zookeeper                   | [2023-10-01 14:41:19,387] INFO Logging initialized @1467ms to org.eclipse.jetty.util.log.Slf4jLog (org.eclipse.jetty.util.log)
zookeeper                   | [2023-10-01 14:41:19,635] WARN o.e.j.s.ServletContextHandler@387a8303{/,null,STOPPED} contextPath ends with /* (org.eclipse.jetty.server.handler.ContextHandler)
zookeeper                   | [2023-10-01 14:41:19,636] WARN Empty contextPath (org.eclipse.jetty.server.handler.ContextHandler)
zookeeper                   | [2023-10-01 14:41:19,692] INFO jetty-9.4.51.v20230217; built: 2023-02-17T08:19:37.309Z; git: b45c405e4544384de066f814ed42ae3dceacdd49; jvm 11.0.20+8-LTS (org.eclipse.jetty.server.Server)
zookeeper                   | [2023-10-01 14:41:19,753] INFO DefaultSessionIdManager workerName=node0 (org.eclipse.jetty.server.session)
zookeeper                   | [2023-10-01 14:41:19,753] INFO No SessionScavenger set, using defaults (org.eclipse.jetty.server.session)
zookeeper                   | [2023-10-01 14:41:19,756] INFO node0 Scavenging every 600000ms (org.eclipse.jetty.server.session)
zookeeper                   | [2023-10-01 14:41:19,761] WARN ServletContext@o.e.j.s.ServletContextHandler@387a8303{/,null,STARTING} has uncovered http methods for path: /* (org.eclipse.jetty.security.SecurityHandler)
zookeeper                   | [2023-10-01 14:41:19,778] INFO Started o.e.j.s.ServletContextHandler@387a8303{/,null,AVAILABLE} (org.eclipse.jetty.server.handler.ContextHandler)
zookeeper                   | [2023-10-01 14:41:19,800] INFO Started ServerConnector@7ce026d3{HTTP/1.1, (http/1.1)}{0.0.0.0:8080} (org.eclipse.jetty.server.AbstractConnector)
zookeeper                   | [2023-10-01 14:41:19,800] INFO Started @1880ms (org.eclipse.jetty.server.Server)
zookeeper                   | [2023-10-01 14:41:19,800] INFO Started AdminServer on address 0.0.0.0, port 8080 and command URL /commands (org.apache.zookeeper.server.admin.JettyAdminServer)
zookeeper                   | [2023-10-01 14:41:19,807] INFO Using org.apache.zookeeper.server.NIOServerCnxnFactory as server connection factory (org.apache.zookeeper.server.ServerCnxnFactory)
zookeeper                   | [2023-10-01 14:41:19,808] WARN maxCnxns is not configured, using default value 0. (org.apache.zookeeper.server.ServerCnxnFactory)
zookeeper                   | [2023-10-01 14:41:19,809] INFO Configuring NIO connection handler with 10s sessionless connection timeout, 1 selector thread(s), 4 worker threads, and 64 kB direct buffers. (org.apache.zookeeper.server.NIOServerCnxnFactory)
zookeeper                   | [2023-10-01 14:41:19,811] INFO binding to port 0.0.0.0/0.0.0.0:2181 (org.apache.zookeeper.server.NIOServerCnxnFactory)
zookeeper                   | [2023-10-01 14:41:19,836] INFO Using org.apache.zookeeper.server.watch.WatchManager as watch manager (org.apache.zookeeper.server.watch.WatchManagerFactory)
zookeeper                   | [2023-10-01 14:41:19,837] INFO Using org.apache.zookeeper.server.watch.WatchManager as watch manager (org.apache.zookeeper.server.watch.WatchManagerFactory)
zookeeper                   | [2023-10-01 14:41:19,839] INFO zookeeper.snapshotSizeFactor = 0.33 (org.apache.zookeeper.server.ZKDatabase)
zookeeper                   | [2023-10-01 14:41:19,840] INFO zookeeper.commitLogCount=500 (org.apache.zookeeper.server.ZKDatabase)
zookeeper                   | [2023-10-01 14:41:19,849] INFO zookeeper.snapshot.compression.method = CHECKED (org.apache.zookeeper.server.persistence.SnapStream)
zookeeper                   | [2023-10-01 14:41:19,853] INFO Snapshotting: 0x0 to /var/lib/zookeeper/data/version-2/snapshot.0 (org.apache.zookeeper.server.persistence.FileTxnSnapLog)
zookeeper                   | [2023-10-01 14:41:19,856] INFO Snapshot loaded in 16 ms, highest zxid is 0x0, digest is 1371985504 (org.apache.zookeeper.server.ZKDatabase)
zookeeper                   | [2023-10-01 14:41:19,856] INFO Snapshotting: 0x0 to /var/lib/zookeeper/data/version-2/snapshot.0 (org.apache.zookeeper.server.persistence.FileTxnSnapLog)
zookeeper                   | [2023-10-01 14:41:19,858] INFO Snapshot taken in 2 ms (org.apache.zookeeper.server.ZooKeeperServer)
zookeeper                   | [2023-10-01 14:41:19,871] INFO zookeeper.request_throttler.shutdownTimeout = 10000 (org.apache.zookeeper.server.RequestThrottler)
zookeeper                   | [2023-10-01 14:41:19,872] INFO PrepRequestProcessor (sid:0) started, reconfigEnabled=false (org.apache.zookeeper.server.PrepRequestProcessor)
kafka                       | [2023-10-01 14:41:19,921] INFO Client environment:zookeeper.version=3.6.3--6401e4ad2087061bc6b9f80dec2d69f2e3c8660a, built on 04/08/2021 16:35 GMT (org.apache.zookeeper.ZooKeeper)
kafka                       | [2023-10-01 14:41:19,925] INFO Client environment:host.name=56efb419ec87 (org.apache.zookeeper.ZooKeeper)
kafka                       | [2023-10-01 14:41:19,925] INFO Client environment:java.version=11.0.20 (org.apache.zookeeper.ZooKeeper)
kafka                       | [2023-10-01 14:41:19,926] INFO Client environment:java.vendor=Azul Systems, Inc. (org.apache.zookeeper.ZooKeeper)
kafka                       | [2023-10-01 14:41:19,927] INFO Client environment:java.home=/usr/lib/jvm/java-11-zulu-openjdk-ca (org.apache.zookeeper.ZooKeeper)
zookeeper                   | [2023-10-01 14:41:19,927] INFO Using checkIntervalMs=60000 maxPerMinute=10000 maxNeverUsedIntervalMs=0 (org.apache.zookeeper.server.ContainerManager)
kafka                       | [2023-10-01 14:41:19,927] INFO Client environment:java.class.path=/usr/share/java/cp-base-new/jackson-databind-2.14.2.jar:/usr/share/java/cp-base-new/jmx_prometheus_javaagent-0.18.0.jar:/usr/share/java/cp-base-new/scala-reflect-2.13.10.jar:/usr/share/java/cp-base-new/metrics-core-4.1.12.1.jar:/usr/share/java/cp-base-new/jackson-module-scala_2.13-2.14.2.jar:/usr/share/java/cp-base-new/scala-library-2.13.10.jar:/usr/share/java/cp-base-new/logredactor-metrics-1.0.12.jar:/usr/share/java/cp-base-new/jackson-dataformat-csv-2.14.2.jar:/usr/share/java/cp-base-new/disk-usage-agent-7.3.5.jar:/usr/share/java/cp-base-new/kafka-storage-api-7.3.5-ccs.jar:/usr/share/java/cp-base-new/argparse4j-0.7.0.jar:/usr/share/java/cp-base-new/scala-collection-compat_2.13-2.6.0.jar:/usr/share/java/cp-base-new/jose4j-0.9.3.jar:/usr/share/java/cp-base-new/jackson-datatype-jdk8-2.14.2.jar:/usr/share/java/cp-base-new/jopt-simple-5.0.4.jar:/usr/share/java/cp-base-new/scala-logging_2.13-3.9.4.jar:/usr/share/java/cp-base-new/kafka-server-common-7.3.5-ccs.jar:/usr/share/java/cp-base-new/lz4-java-1.8.0.jar:/usr/share/java/cp-base-new/minimal-json-0.9.5.jar:/usr/share/java/cp-base-new/utility-belt-7.3.5.jar:/usr/share/java/cp-base-new/slf4j-api-1.7.36.jar:/usr/share/java/cp-base-new/re2j-1.6.jar:/usr/share/java/cp-base-new/paranamer-2.8.jar:/usr/share/java/cp-base-new/kafka_2.13-7.3.5-ccs.jar:/usr/share/java/cp-base-new/metrics-core-2.2.0.jar:/usr/share/java/cp-base-new/zookeeper-3.6.3.jar:/usr/share/java/cp-base-new/jolokia-jvm-1.7.1.jar:/usr/share/java/cp-base-new/logredactor-1.0.12.jar:/usr/share/java/cp-base-new/kafka-raft-7.3.5-ccs.jar:/usr/share/java/cp-base-new/kafka-storage-7.3.5-ccs.jar:/usr/share/java/cp-base-new/jackson-dataformat-yaml-2.14.2.jar:/usr/share/java/cp-base-new/reload4j-1.2.19.jar:/usr/share/java/cp-base-new/jackson-annotations-2.14.2.jar:/usr/share/java/cp-base-new/zstd-jni-1.5.2-1.jar:/usr/share/java/cp-base-new/kafka-metadata-7.3.5-ccs.jar:/usr/share/java/cp-base-new/snappy-java-1.1.10.1.jar:/usr/share/java/cp-base-new/snakeyaml-2.0.jar:/usr/share/java/cp-base-new/kafka-clients-7.3.5-ccs.jar:/usr/share/java/cp-base-new/jolokia-core-1.7.1.jar:/usr/share/java/cp-base-new/json-simple-1.1.1.jar:/usr/share/java/cp-base-new/scala-java8-compat_2.13-1.0.2.jar:/usr/share/java/cp-base-new/zookeeper-jute-3.6.3.jar:/usr/share/java/cp-base-new/common-utils-7.3.5.jar:/usr/share/java/cp-base-new/gson-2.9.0.jar:/usr/share/java/cp-base-new/commons-cli-1.4.jar:/usr/share/java/cp-base-new/audience-annotations-0.5.0.jar:/usr/share/java/cp-base-new/slf4j-reload4j-1.7.36.jar:/usr/share/java/cp-base-new/jackson-core-2.14.2.jar (org.apache.zookeeper.ZooKeeper)
kafka                       | [2023-10-01 14:41:19,928] INFO Client environment:java.library.path=/usr/java/packages/lib:/usr/lib64:/lib64:/lib:/usr/lib (org.apache.zookeeper.ZooKeeper)
kafka                       | [2023-10-01 14:41:19,928] INFO Client environment:java.io.tmpdir=/tmp (org.apache.zookeeper.ZooKeeper)
kafka                       | [2023-10-01 14:41:19,928] INFO Client environment:java.compiler=<NA> (org.apache.zookeeper.ZooKeeper)
kafka                       | [2023-10-01 14:41:19,929] INFO Client environment:os.name=Linux (org.apache.zookeeper.ZooKeeper)
kafka                       | [2023-10-01 14:41:19,929] INFO Client environment:os.arch=amd64 (org.apache.zookeeper.ZooKeeper)
kafka                       | [2023-10-01 14:41:19,929] INFO Client environment:os.version=6.2.0-1012-azure (org.apache.zookeeper.ZooKeeper)
kafka                       | [2023-10-01 14:41:19,929] INFO Client environment:user.name=appuser (org.apache.zookeeper.ZooKeeper)
kafka                       | [2023-10-01 14:41:19,929] INFO Client environment:user.home=/home/appuser (org.apache.zookeeper.ZooKeeper)
kafka                       | [2023-10-01 14:41:19,929] INFO Client environment:user.dir=/home/appuser (org.apache.zookeeper.ZooKeeper)
kafka                       | [2023-10-01 14:41:19,929] INFO Client environment:os.memory.free=104MB (org.apache.zookeeper.ZooKeeper)
kafka                       | [2023-10-01 14:41:19,929] INFO Client environment:os.memory.max=1732MB (org.apache.zookeeper.ZooKeeper)
kafka                       | [2023-10-01 14:41:19,929] INFO Client environment:os.memory.total=110MB (org.apache.zookeeper.ZooKeeper)
zookeeper                   | [2023-10-01 14:41:19,930] INFO ZooKeeper audit is disabled. (org.apache.zookeeper.audit.ZKAuditProvider)
kafka                       | [2023-10-01 14:41:19,938] INFO Initiating client connection, connectString=zookeeper:2181 sessionTimeout=40000 watcher=io.confluent.admin.utils.ZookeeperConnectionWatcher@797badd3 (org.apache.zookeeper.ZooKeeper)
kafka                       | [2023-10-01 14:41:19,947] INFO Setting -D jdk.tls.rejectClientInitiatedRenegotiation=true to disable client-initiated TLS renegotiation (org.apache.zookeeper.common.X509Util)
kafka                       | [2023-10-01 14:41:19,959] INFO jute.maxbuffer value is 1048575 Bytes (org.apache.zookeeper.ClientCnxnSocket)
kafka                       | [2023-10-01 14:41:19,966] INFO zookeeper.request.timeout value is 0. feature enabled=false (org.apache.zookeeper.ClientCnxn)
kafka                       | [2023-10-01 14:41:19,982] INFO Opening socket connection to server zookeeper/172.18.0.3:2181. (org.apache.zookeeper.ClientCnxn)
kafka                       | [2023-10-01 14:41:19,982] INFO SASL config status: Will not attempt to authenticate using SASL (unknown error) (org.apache.zookeeper.ClientCnxn)
kafka                       | [2023-10-01 14:41:19,992] INFO Socket connection established, initiating session, client: /172.18.0.4:50818, server: zookeeper/172.18.0.3:2181 (org.apache.zookeeper.ClientCnxn)
zookeeper                   | [2023-10-01 14:41:20,004] INFO Creating new log file: log.1 (org.apache.zookeeper.server.persistence.FileTxnLog)
kafka                       | [2023-10-01 14:41:20,016] INFO Session establishment complete on server zookeeper/172.18.0.3:2181, session id = 0x1000002dbfe0000, negotiated timeout = 40000 (org.apache.zookeeper.ClientCnxn)
kafka                       | [2023-10-01 14:41:20,147] INFO Session: 0x1000002dbfe0000 closed (org.apache.zookeeper.ZooKeeper)
kafka                       | [2023-10-01 14:41:20,147] INFO EventThread shut down for session: 0x1000002dbfe0000 (org.apache.zookeeper.ClientCnxn)
kafka                       | Using log4j config /etc/kafka/log4j.properties
kafka                       | ===> Launching ... 
kafka                       | ===> Launching kafka ... 
kafka                       | [2023-10-01 14:41:20,948] INFO Registered kafka:type=kafka.Log4jController MBean (kafka.utils.Log4jControllerRegistration$)
kafka                       | [2023-10-01 14:41:21,281] INFO Setting -D jdk.tls.rejectClientInitiatedRenegotiation=true to disable client-initiated TLS renegotiation (org.apache.zookeeper.common.X509Util)
kafka                       | [2023-10-01 14:41:21,392] INFO Registered signal handlers for TERM, INT, HUP (org.apache.kafka.common.utils.LoggingSignalHandler)
kafka                       | [2023-10-01 14:41:21,394] INFO starting (kafka.server.KafkaServer)
kafka                       | [2023-10-01 14:41:21,395] INFO Connecting to zookeeper on zookeeper:2181 (kafka.server.KafkaServer)
kafka                       | [2023-10-01 14:41:21,412] INFO [ZooKeeperClient Kafka server] Initializing a new session to zookeeper:2181. (kafka.zookeeper.ZooKeeperClient)
kafka                       | [2023-10-01 14:41:21,417] INFO Client environment:zookeeper.version=3.6.4--d65253dcf68e9097c6e95a126463fd5fdeb4521c, built on 12/18/2022 18:10 GMT (org.apache.zookeeper.ZooKeeper)
kafka                       | [2023-10-01 14:41:21,417] INFO Client environment:host.name=56efb419ec87 (org.apache.zookeeper.ZooKeeper)
kafka                       | [2023-10-01 14:41:21,417] INFO Client environment:java.version=11.0.20 (org.apache.zookeeper.ZooKeeper)
kafka                       | [2023-10-01 14:41:21,417] INFO Client environment:java.vendor=Azul Systems, Inc. (org.apache.zookeeper.ZooKeeper)
kafka                       | [2023-10-01 14:41:21,417] INFO Client environment:java.home=/usr/lib/jvm/java-11-zulu-openjdk-ca (org.apache.zookeeper.ZooKeeper)
kafka                       | [2023-10-01 14:41:21,417] INFO Client environment:java.class.path=/usr/bin/../share/java/kafka/netty-codec-4.1.96.Final.jar:/usr/bin/../share/java/kafka/jackson-databind-2.13.4.2.jar:/usr/bin/../share/java/kafka/javassist-3.27.0-GA.jar:/usr/bin/../share/java/kafka/connect-mirror-7.3.5-ccs.jar:/usr/bin/../share/java/kafka/jakarta.annotation-api-1.3.5.jar:/usr/bin/../share/java/kafka/netty-resolver-4.1.96.Final.jar:/usr/bin/../share/java/kafka/scala-reflect-2.13.10.jar:/usr/bin/../share/java/kafka/metrics-core-4.1.12.1.jar:/usr/bin/../share/java/kafka/jetty-util-9.4.51.v20230217.jar:/usr/bin/../share/java/kafka/plexus-utils-3.3.0.jar:/usr/bin/../share/java/kafka/activation-1.1.1.jar:/usr/bin/../share/java/kafka/netty-handler-4.1.96.Final.jar:/usr/bin/../share/java/kafka/scala-library-2.13.10.jar:/usr/bin/../share/java/kafka/javax.servlet-api-3.1.0.jar:/usr/bin/../share/java/kafka/rocksdbjni-7.1.2.jar:/usr/bin/../share/java/kafka/kafka-streams-examples-7.3.5-ccs.jar:/usr/bin/../share/java/kafka/jackson-core-2.13.4.jar:/usr/bin/../share/java/kafka/zookeeper-3.6.4.jar:/usr/bin/../share/java/kafka/jackson-datatype-jdk8-2.13.4.jar:/usr/bin/../share/java/kafka/commons-lang3-3.8.1.jar:/usr/bin/../share/java/kafka/kafka-storage-api-7.3.5-ccs.jar:/usr/bin/../share/java/kafka/netty-common-4.1.96.Final.jar:/usr/bin/../share/java/kafka/argparse4j-0.7.0.jar:/usr/bin/../share/java/kafka/scala-collection-compat_2.13-2.6.0.jar:/usr/bin/../share/java/kafka/netty-transport-classes-epoll-4.1.96.Final.jar:/usr/bin/../share/java/kafka/netty-buffer-4.1.96.Final.jar:/usr/bin/../share/java/kafka/jose4j-0.9.3.jar:/usr/bin/../share/java/kafka/javax.ws.rs-api-2.1.1.jar:/usr/bin/../share/java/kafka/jersey-server-2.34.jar:/usr/bin/../share/java/kafka/kafka-shell-7.3.5-ccs.jar:/usr/bin/../share/java/kafka/connect-runtime-7.3.5-ccs.jar:/usr/bin/../share/java/kafka/connect-mirror-client-7.3.5-ccs.jar:/usr/bin/../share/java/kafka/jackson-module-jaxb-annotations-2.13.4.jar:/usr/bin/../share/java/kafka/jackson-jaxrs-base-2.13.4.jar:/usr/bin/../share/java/kafka/netty-transport-native-epoll-4.1.96.Final.jar:/usr/bin/../share/java/kafka/jetty-io-9.4.51.v20230217.jar:/usr/bin/../share/java/kafka/hk2-api-2.6.1.jar:/usr/bin/../share/java/kafka/swagger-annotations-2.2.0.jar:/usr/bin/../share/java/kafka/jakarta.activation-api-1.2.2.jar:/usr/bin/../share/java/kafka/jopt-simple-5.0.4.jar:/usr/bin/../share/java/kafka/aopalliance-repackaged-2.6.1.jar:/usr/bin/../share/java/kafka/scala-logging_2.13-3.9.4.jar:/usr/bin/../share/java/kafka/jakarta.ws.rs-api-2.1.6.jar:/usr/bin/../share/java/kafka/connect-transforms-7.3.5-ccs.jar:/usr/bin/../share/java/kafka/kafka-server-common-7.3.5-ccs.jar:/usr/bin/../share/java/kafka/lz4-java-1.8.0.jar:/usr/bin/../share/java/kafka/jackson-jaxrs-json-provider-2.13.4.jar:/usr/bin/../share/java/kafka/kafka-streams-test-utils-7.3.5-ccs.jar:/usr/bin/../share/java/kafka/slf4j-api-1.7.36.jar:/usr/bin/../share/java/kafka/connect-json-7.3.5-ccs.jar:/usr/bin/../share/java/kafka/paranamer-2.8.jar:/usr/bin/../share/java/kafka/kafka_2.13-7.3.5-ccs.jar:/usr/bin/../share/java/kafka/metrics-core-2.2.0.jar:/usr/bin/../share/java/kafka/jersey-common-2.34.jar:/usr/bin/../share/java/kafka/maven-artifact-3.8.4.jar:/usr/bin/../share/java/kafka/jakarta.xml.bind-api-2.3.3.jar:/usr/bin/../share/java/kafka/jersey-container-servlet-core-2.34.jar:/usr/bin/../share/java/kafka/jackson-module-scala_2.13-2.13.4.jar:/usr/bin/../share/java/kafka/reflections-0.9.12.jar:/usr/bin/../share/java/kafka/kafka-streams-scala_2.13-7.3.5-ccs.jar:/usr/bin/../share/java/kafka/kafka-raft-7.3.5-ccs.jar:/usr/bin/../share/java/kafka/hk2-locator-2.6.1.jar:/usr/bin/../share/java/kafka/audience-annotations-0.13.0.jar:/usr/bin/../share/java/kafka/kafka-storage-7.3.5-ccs.jar:/usr/bin/../share/java/kafka/jetty-security-9.4.51.v20230217.jar:/usr/bin/../share/java/kafka/netty-transport-native-unix-common-4.1.96.Final.jar:/usr/bin/../share/java/kafka/jetty-continuation-9.4.51.v20230217.jar:/usr/bin/../share/java/kafka/zookeeper-jute-3.6.4.jar:/usr/bin/../share/java/kafka/jetty-http-9.4.51.v20230217.jar:/usr/bin/../share/java/kafka/jersey-container-servlet-2.34.jar:/usr/bin/../share/java/kafka/jetty-server-9.4.51.v20230217.jar:/usr/bin/../share/java/kafka/reload4j-1.2.19.jar:/usr/bin/../share/java/kafka/jackson-dataformat-csv-2.13.4.jar:/usr/bin/../share/java/kafka/zstd-jni-1.5.2-1.jar:/usr/bin/../share/java/kafka/jetty-servlets-9.4.51.v20230217.jar:/usr/bin/../share/java/kafka/jetty-servlet-9.4.51.v20230217.jar:/usr/bin/../share/java/kafka/kafka-metadata-7.3.5-ccs.jar:/usr/bin/../share/java/kafka/jersey-hk2-2.34.jar:/usr/bin/../share/java/kafka/snappy-java-1.1.10.1.jar:/usr/bin/../share/java/kafka/hk2-utils-2.6.1.jar:/usr/bin/../share/java/kafka/commons-lang3-3.12.0.jar:/usr/bin/../share/java/kafka/jersey-client-2.34.jar:/usr/bin/../share/java/kafka/jetty-util-ajax-9.4.51.v20230217.jar:/usr/bin/../share/java/kafka/kafka-tools-7.3.5-ccs.jar:/usr/bin/../share/java/kafka/jline-3.21.0.jar:/usr/bin/../share/java/kafka/jakarta.inject-2.6.1.jar:/usr/bin/../share/java/kafka/jaxb-api-2.3.0.jar:/usr/bin/../share/java/kafka/jackson-annotations-2.13.4.jar:/usr/bin/../share/java/kafka/jakarta.validation-api-2.0.2.jar:/usr/bin/../share/java/kafka/trogdor-7.3.5-ccs.jar:/usr/bin/../share/java/kafka/kafka-clients-7.3.5-ccs.jar:/usr/bin/../share/java/kafka/kafka-log4j-appender-7.3.5-ccs.jar:/usr/bin/../share/java/kafka/netty-transport-4.1.96.Final.jar:/usr/bin/../share/java/kafka/kafka.jar:/usr/bin/../share/java/kafka/scala-java8-compat_2.13-1.0.2.jar:/usr/bin/../share/java/kafka/connect-api-7.3.5-ccs.jar:/usr/bin/../share/java/kafka/kafka-streams-7.3.5-ccs.jar:/usr/bin/../share/java/kafka/osgi-resource-locator-1.0.3.jar:/usr/bin/../share/java/kafka/jetty-client-9.4.51.v20230217.jar:/usr/bin/../share/java/kafka/commons-cli-1.4.jar:/usr/bin/../share/java/kafka/slf4j-reload4j-1.7.36.jar:/usr/bin/../share/java/kafka/connect-basic-auth-extension-7.3.5-ccs.jar:/usr/bin/../share/java/confluent-telemetry/* (org.apache.zookeeper.ZooKeeper)
kafka                       | [2023-10-01 14:41:21,417] INFO Client environment:java.library.path=/usr/java/packages/lib:/usr/lib64:/lib64:/lib:/usr/lib (org.apache.zookeeper.ZooKeeper)
kafka                       | [2023-10-01 14:41:21,417] INFO Client environment:java.io.tmpdir=/tmp (org.apache.zookeeper.ZooKeeper)
kafka                       | [2023-10-01 14:41:21,417] INFO Client environment:java.compiler=<NA> (org.apache.zookeeper.ZooKeeper)
kafka                       | [2023-10-01 14:41:21,417] INFO Client environment:os.name=Linux (org.apache.zookeeper.ZooKeeper)
kafka                       | [2023-10-01 14:41:21,417] INFO Client environment:os.arch=amd64 (org.apache.zookeeper.ZooKeeper)
kafka                       | [2023-10-01 14:41:21,417] INFO Client environment:os.version=6.2.0-1012-azure (org.apache.zookeeper.ZooKeeper)
kafka                       | [2023-10-01 14:41:21,417] INFO Client environment:user.name=appuser (org.apache.zookeeper.ZooKeeper)
kafka                       | [2023-10-01 14:41:21,417] INFO Client environment:user.home=/home/appuser (org.apache.zookeeper.ZooKeeper)
kafka                       | [2023-10-01 14:41:21,417] INFO Client environment:user.dir=/home/appuser (org.apache.zookeeper.ZooKeeper)
kafka                       | [2023-10-01 14:41:21,417] INFO Client environment:os.memory.free=1010MB (org.apache.zookeeper.ZooKeeper)
kafka                       | [2023-10-01 14:41:21,417] INFO Client environment:os.memory.max=1024MB (org.apache.zookeeper.ZooKeeper)
kafka                       | [2023-10-01 14:41:21,417] INFO Client environment:os.memory.total=1024MB (org.apache.zookeeper.ZooKeeper)
kafka                       | [2023-10-01 14:41:21,422] INFO Initiating client connection, connectString=zookeeper:2181 sessionTimeout=18000 watcher=kafka.zookeeper.ZooKeeperClient$ZooKeeperClientWatcher$@67304a40 (org.apache.zookeeper.ZooKeeper)
kafka                       | [2023-10-01 14:41:21,429] INFO jute.maxbuffer value is 4194304 Bytes (org.apache.zookeeper.ClientCnxnSocket)
kafka                       | [2023-10-01 14:41:21,435] INFO zookeeper.request.timeout value is 0. feature enabled=false (org.apache.zookeeper.ClientCnxn)
kafka                       | [2023-10-01 14:41:21,438] INFO [ZooKeeperClient Kafka server] Waiting until connected. (kafka.zookeeper.ZooKeeperClient)
kafka                       | [2023-10-01 14:41:21,441] INFO Opening socket connection to server zookeeper/172.18.0.3:2181. (org.apache.zookeeper.ClientCnxn)
kafka                       | [2023-10-01 14:41:21,447] INFO Socket connection established, initiating session, client: /172.18.0.4:50828, server: zookeeper/172.18.0.3:2181 (org.apache.zookeeper.ClientCnxn)
kafka                       | [2023-10-01 14:41:21,456] INFO Session establishment complete on server zookeeper/172.18.0.3:2181, session id = 0x1000002dbfe0001, negotiated timeout = 18000 (org.apache.zookeeper.ClientCnxn)
kafka                       | [2023-10-01 14:41:21,460] INFO [ZooKeeperClient Kafka server] Connected. (kafka.zookeeper.ZooKeeperClient)
kafka                       | [2023-10-01 14:41:21,768] INFO Cluster ID = iQ9qSqF6TbWO9XLlfB04BA (kafka.server.KafkaServer)
kafka                       | [2023-10-01 14:41:21,773] WARN No meta.properties file under dir /var/lib/kafka/data/meta.properties (kafka.server.BrokerMetadataCheckpoint)
kafka                       | [2023-10-01 14:41:21,833] INFO KafkaConfig values: 
kafka                       | 	advertised.listeners = PLAINTEXT://kafka:29092,PLAINTEXT_HOST://localhost:9092
kafka                       | 	alter.config.policy.class.name = null
kafka                       | 	alter.log.dirs.replication.quota.window.num = 11
kafka                       | 	alter.log.dirs.replication.quota.window.size.seconds = 1
kafka                       | 	authorizer.class.name = 
kafka                       | 	auto.create.topics.enable = true
kafka                       | 	auto.leader.rebalance.enable = true
kafka                       | 	background.threads = 10
kafka                       | 	broker.heartbeat.interval.ms = 2000
kafka                       | 	broker.id = -1
kafka                       | 	broker.id.generation.enable = true
kafka                       | 	broker.rack = null
kafka                       | 	broker.session.timeout.ms = 9000
kafka                       | 	client.quota.callback.class = null
kafka                       | 	compression.type = producer
kafka                       | 	connection.failed.authentication.delay.ms = 100
kafka                       | 	connections.max.idle.ms = 600000
kafka                       | 	connections.max.reauth.ms = 0
kafka                       | 	control.plane.listener.name = null
kafka                       | 	controlled.shutdown.enable = true
kafka                       | 	controlled.shutdown.max.retries = 3
kafka                       | 	controlled.shutdown.retry.backoff.ms = 5000
kafka                       | 	controller.listener.names = null
kafka                       | 	controller.quorum.append.linger.ms = 25
kafka                       | 	controller.quorum.election.backoff.max.ms = 1000
kafka                       | 	controller.quorum.election.timeout.ms = 1000
kafka                       | 	controller.quorum.fetch.timeout.ms = 2000
kafka                       | 	controller.quorum.request.timeout.ms = 2000
kafka                       | 	controller.quorum.retry.backoff.ms = 20
kafka                       | 	controller.quorum.voters = []
kafka                       | 	controller.quota.window.num = 11
kafka                       | 	controller.quota.window.size.seconds = 1
kafka                       | 	controller.socket.timeout.ms = 30000
kafka                       | 	create.topic.policy.class.name = null
kafka                       | 	default.replication.factor = 1
kafka                       | 	delegation.token.expiry.check.interval.ms = 3600000
kafka                       | 	delegation.token.expiry.time.ms = 86400000
kafka                       | 	delegation.token.master.key = null
kafka                       | 	delegation.token.max.lifetime.ms = 604800000
kafka                       | 	delegation.token.secret.key = null
kafka                       | 	delete.records.purgatory.purge.interval.requests = 1
kafka                       | 	delete.topic.enable = true
kafka                       | 	early.start.listeners = null
kafka                       | 	fetch.max.bytes = 57671680
kafka                       | 	fetch.purgatory.purge.interval.requests = 1000
kafka                       | 	group.initial.rebalance.delay.ms = 3000
kafka                       | 	group.max.session.timeout.ms = 1800000
kafka                       | 	group.max.size = 2147483647
kafka                       | 	group.min.session.timeout.ms = 6000
kafka                       | 	initial.broker.registration.timeout.ms = 60000
kafka                       | 	inter.broker.listener.name = PLAINTEXT
kafka                       | 	inter.broker.protocol.version = 3.3-IV3
kafka                       | 	kafka.metrics.polling.interval.secs = 10
kafka                       | 	kafka.metrics.reporters = []
kafka                       | 	leader.imbalance.check.interval.seconds = 300
kafka                       | 	leader.imbalance.per.broker.percentage = 10
kafka                       | 	listener.security.protocol.map = PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT
kafka                       | 	listeners = PLAINTEXT://0.0.0.0:29092,PLAINTEXT_HOST://0.0.0.0:9092
kafka                       | 	log.cleaner.backoff.ms = 15000
kafka                       | 	log.cleaner.dedupe.buffer.size = 134217728
kafka                       | 	log.cleaner.delete.retention.ms = 86400000
kafka                       | 	log.cleaner.enable = true
kafka                       | 	log.cleaner.io.buffer.load.factor = 0.9
kafka                       | 	log.cleaner.io.buffer.size = 524288
kafka                       | 	log.cleaner.io.max.bytes.per.second = 1.7976931348623157E308
kafka                       | 	log.cleaner.max.compaction.lag.ms = 9223372036854775807
kafka                       | 	log.cleaner.min.cleanable.ratio = 0.5
kafka                       | 	log.cleaner.min.compaction.lag.ms = 0
kafka                       | 	log.cleaner.threads = 1
kafka                       | 	log.cleanup.policy = [delete]
kafka                       | 	log.dir = /tmp/kafka-logs
kafka                       | 	log.dirs = /var/lib/kafka/data
kafka                       | 	log.flush.interval.messages = 9223372036854775807
kafka                       | 	log.flush.interval.ms = null
kafka                       | 	log.flush.offset.checkpoint.interval.ms = 60000
kafka                       | 	log.flush.scheduler.interval.ms = 9223372036854775807
kafka                       | 	log.flush.start.offset.checkpoint.interval.ms = 60000
kafka                       | 	log.index.interval.bytes = 4096
kafka                       | 	log.index.size.max.bytes = 10485760
kafka                       | 	log.message.downconversion.enable = true
kafka                       | 	log.message.format.version = 3.0-IV1
kafka                       | 	log.message.timestamp.difference.max.ms = 9223372036854775807
kafka                       | 	log.message.timestamp.type = CreateTime
kafka                       | 	log.preallocate = false
kafka                       | 	log.retention.bytes = -1
kafka                       | 	log.retention.check.interval.ms = 300000
kafka                       | 	log.retention.hours = 168
kafka                       | 	log.retention.minutes = null
kafka                       | 	log.retention.ms = null
kafka                       | 	log.roll.hours = 168
kafka                       | 	log.roll.jitter.hours = 0
kafka                       | 	log.roll.jitter.ms = null
kafka                       | 	log.roll.ms = null
kafka                       | 	log.segment.bytes = 1073741824
kafka                       | 	log.segment.delete.delay.ms = 60000
kafka                       | 	max.connection.creation.rate = 2147483647
kafka                       | 	max.connections = 2147483647
kafka                       | 	max.connections.per.ip = 2147483647
kafka                       | 	max.connections.per.ip.overrides = 
kafka                       | 	max.incremental.fetch.session.cache.slots = 1000
kafka                       | 	message.max.bytes = 1048588
kafka                       | 	metadata.log.dir = null
kafka                       | 	metadata.log.max.record.bytes.between.snapshots = 20971520
kafka                       | 	metadata.log.segment.bytes = 1073741824
kafka                       | 	metadata.log.segment.min.bytes = 8388608
kafka                       | 	metadata.log.segment.ms = 604800000
kafka                       | 	metadata.max.idle.interval.ms = 500
kafka                       | 	metadata.max.retention.bytes = -1
kafka                       | 	metadata.max.retention.ms = 604800000
kafka                       | 	metric.reporters = []
kafka                       | 	metrics.num.samples = 2
kafka                       | 	metrics.recording.level = INFO
kafka                       | 	metrics.sample.window.ms = 30000
kafka                       | 	min.insync.replicas = 1
kafka                       | 	node.id = -1
kafka                       | 	num.io.threads = 8
kafka                       | 	num.network.threads = 3
kafka                       | 	num.partitions = 1
kafka                       | 	num.recovery.threads.per.data.dir = 1
kafka                       | 	num.replica.alter.log.dirs.threads = null
kafka                       | 	num.replica.fetchers = 1
kafka                       | 	offset.metadata.max.bytes = 4096
kafka                       | 	offsets.commit.required.acks = -1
kafka                       | 	offsets.commit.timeout.ms = 5000
kafka                       | 	offsets.load.buffer.size = 5242880
kafka                       | 	offsets.retention.check.interval.ms = 600000
kafka                       | 	offsets.retention.minutes = 10080
kafka                       | 	offsets.topic.compression.codec = 0
kafka                       | 	offsets.topic.num.partitions = 50
kafka                       | 	offsets.topic.replication.factor = 1
kafka                       | 	offsets.topic.segment.bytes = 104857600
kafka                       | 	password.encoder.cipher.algorithm = AES/CBC/PKCS5Padding
kafka                       | 	password.encoder.iterations = 4096
kafka                       | 	password.encoder.key.length = 128
kafka                       | 	password.encoder.keyfactory.algorithm = null
kafka                       | 	password.encoder.old.secret = null
kafka                       | 	password.encoder.secret = null
kafka                       | 	principal.builder.class = class org.apache.kafka.common.security.authenticator.DefaultKafkaPrincipalBuilder
kafka                       | 	process.roles = []
kafka                       | 	producer.purgatory.purge.interval.requests = 1000
kafka                       | 	queued.max.request.bytes = -1
kafka                       | 	queued.max.requests = 500
kafka                       | 	quota.window.num = 11
kafka                       | 	quota.window.size.seconds = 1
kafka                       | 	remote.log.index.file.cache.total.size.bytes = 1073741824
kafka                       | 	remote.log.manager.task.interval.ms = 30000
kafka                       | 	remote.log.manager.task.retry.backoff.max.ms = 30000
kafka                       | 	remote.log.manager.task.retry.backoff.ms = 500
kafka                       | 	remote.log.manager.task.retry.jitter = 0.2
kafka                       | 	remote.log.manager.thread.pool.size = 10
kafka                       | 	remote.log.metadata.manager.class.name = null
kafka                       | 	remote.log.metadata.manager.class.path = null
kafka                       | 	remote.log.metadata.manager.impl.prefix = null
kafka                       | 	remote.log.metadata.manager.listener.name = null
kafka                       | 	remote.log.reader.max.pending.tasks = 100
kafka                       | 	remote.log.reader.threads = 10
kafka                       | 	remote.log.storage.manager.class.name = null
kafka                       | 	remote.log.storage.manager.class.path = null
kafka                       | 	remote.log.storage.manager.impl.prefix = null
kafka                       | 	remote.log.storage.system.enable = false
kafka                       | 	replica.fetch.backoff.ms = 1000
kafka                       | 	replica.fetch.max.bytes = 1048576
kafka                       | 	replica.fetch.min.bytes = 1
kafka                       | 	replica.fetch.response.max.bytes = 10485760
kafka                       | 	replica.fetch.wait.max.ms = 500
kafka                       | 	replica.high.watermark.checkpoint.interval.ms = 5000
kafka                       | 	replica.lag.time.max.ms = 30000
kafka                       | 	replica.selector.class = null
kafka                       | 	replica.socket.receive.buffer.bytes = 65536
kafka                       | 	replica.socket.timeout.ms = 30000
kafka                       | 	replication.quota.window.num = 11
kafka                       | 	replication.quota.window.size.seconds = 1
kafka                       | 	request.timeout.ms = 30000
kafka                       | 	reserved.broker.max.id = 1000
kafka                       | 	sasl.client.callback.handler.class = null
kafka                       | 	sasl.enabled.mechanisms = [GSSAPI]
kafka                       | 	sasl.jaas.config = null
kafka                       | 	sasl.kerberos.kinit.cmd = /usr/bin/kinit
kafka                       | 	sasl.kerberos.min.time.before.relogin = 60000
kafka                       | 	sasl.kerberos.principal.to.local.rules = [DEFAULT]
kafka                       | 	sasl.kerberos.service.name = null
kafka                       | 	sasl.kerberos.ticket.renew.jitter = 0.05
kafka                       | 	sasl.kerberos.ticket.renew.window.factor = 0.8
kafka                       | 	sasl.login.callback.handler.class = null
kafka                       | 	sasl.login.class = null
kafka                       | 	sasl.login.connect.timeout.ms = null
kafka                       | 	sasl.login.read.timeout.ms = null
kafka                       | 	sasl.login.refresh.buffer.seconds = 300
kafka                       | 	sasl.login.refresh.min.period.seconds = 60
kafka                       | 	sasl.login.refresh.window.factor = 0.8
kafka                       | 	sasl.login.refresh.window.jitter = 0.05
kafka                       | 	sasl.login.retry.backoff.max.ms = 10000
kafka                       | 	sasl.login.retry.backoff.ms = 100
kafka                       | 	sasl.mechanism.controller.protocol = GSSAPI
kafka                       | 	sasl.mechanism.inter.broker.protocol = GSSAPI
kafka                       | 	sasl.oauthbearer.clock.skew.seconds = 30
kafka                       | 	sasl.oauthbearer.expected.audience = null
kafka                       | 	sasl.oauthbearer.expected.issuer = null
kafka                       | 	sasl.oauthbearer.jwks.endpoint.refresh.ms = 3600000
kafka                       | 	sasl.oauthbearer.jwks.endpoint.retry.backoff.max.ms = 10000
kafka                       | 	sasl.oauthbearer.jwks.endpoint.retry.backoff.ms = 100
kafka                       | 	sasl.oauthbearer.jwks.endpoint.url = null
kafka                       | 	sasl.oauthbearer.scope.claim.name = scope
kafka                       | 	sasl.oauthbearer.sub.claim.name = sub
kafka                       | 	sasl.oauthbearer.token.endpoint.url = null
kafka                       | 	sasl.server.callback.handler.class = null
kafka                       | 	sasl.server.max.receive.size = 524288
kafka                       | 	security.inter.broker.protocol = PLAINTEXT
kafka                       | 	security.providers = null
kafka                       | 	socket.connection.setup.timeout.max.ms = 30000
kafka                       | 	socket.connection.setup.timeout.ms = 10000
kafka                       | 	socket.listen.backlog.size = 50
kafka                       | 	socket.receive.buffer.bytes = 102400
kafka                       | 	socket.request.max.bytes = 104857600
kafka                       | 	socket.send.buffer.bytes = 102400
kafka                       | 	ssl.cipher.suites = []
kafka                       | 	ssl.client.auth = none
kafka                       | 	ssl.enabled.protocols = [TLSv1.2, TLSv1.3]
kafka                       | 	ssl.endpoint.identification.algorithm = https
kafka                       | 	ssl.engine.factory.class = null
kafka                       | 	ssl.key.password = null
kafka                       | 	ssl.keymanager.algorithm = SunX509
kafka                       | 	ssl.keystore.certificate.chain = null
kafka                       | 	ssl.keystore.key = null
kafka                       | 	ssl.keystore.location = null
kafka                       | 	ssl.keystore.password = null
kafka                       | 	ssl.keystore.type = JKS
kafka                       | 	ssl.principal.mapping.rules = DEFAULT
kafka                       | 	ssl.protocol = TLSv1.3
kafka                       | 	ssl.provider = null
kafka                       | 	ssl.secure.random.implementation = null
kafka                       | 	ssl.trustmanager.algorithm = PKIX
kafka                       | 	ssl.truststore.certificates = null
kafka                       | 	ssl.truststore.location = null
kafka                       | 	ssl.truststore.password = null
kafka                       | 	ssl.truststore.type = JKS
kafka                       | 	transaction.abort.timed.out.transaction.cleanup.interval.ms = 10000
kafka                       | 	transaction.max.timeout.ms = 900000
kafka                       | 	transaction.remove.expired.transaction.cleanup.interval.ms = 3600000
kafka                       | 	transaction.state.log.load.buffer.size = 5242880
kafka                       | 	transaction.state.log.min.isr = 2
kafka                       | 	transaction.state.log.num.partitions = 50
kafka                       | 	transaction.state.log.replication.factor = 3
kafka                       | 	transaction.state.log.segment.bytes = 104857600
kafka                       | 	transactional.id.expiration.ms = 604800000
kafka                       | 	unclean.leader.election.enable = false
kafka                       | 	zookeeper.clientCnxnSocket = null
kafka                       | 	zookeeper.connect = zookeeper:2181
kafka                       | 	zookeeper.connection.timeout.ms = null
kafka                       | 	zookeeper.max.in.flight.requests = 10
kafka                       | 	zookeeper.session.timeout.ms = 18000
kafka                       | 	zookeeper.set.acl = false
kafka                       | 	zookeeper.ssl.cipher.suites = null
kafka                       | 	zookeeper.ssl.client.enable = false
kafka                       | 	zookeeper.ssl.crl.enable = false
kafka                       | 	zookeeper.ssl.enabled.protocols = null
kafka                       | 	zookeeper.ssl.endpoint.identification.algorithm = HTTPS
kafka                       | 	zookeeper.ssl.keystore.location = null
kafka                       | 	zookeeper.ssl.keystore.password = null
kafka                       | 	zookeeper.ssl.keystore.type = null
kafka                       | 	zookeeper.ssl.ocsp.enable = false
kafka                       | 	zookeeper.ssl.protocol = TLSv1.2
kafka                       | 	zookeeper.ssl.truststore.location = null
kafka                       | 	zookeeper.ssl.truststore.password = null
kafka                       | 	zookeeper.ssl.truststore.type = null
kafka                       |  (kafka.server.KafkaConfig)
kafka                       | [2023-10-01 14:41:21,910] INFO [ThrottledChannelReaper-Fetch]: Starting (kafka.server.ClientQuotaManager$ThrottledChannelReaper)
kafka                       | [2023-10-01 14:41:21,911] INFO [ThrottledChannelReaper-Produce]: Starting (kafka.server.ClientQuotaManager$ThrottledChannelReaper)
kafka                       | [2023-10-01 14:41:21,912] INFO [ThrottledChannelReaper-Request]: Starting (kafka.server.ClientQuotaManager$ThrottledChannelReaper)
kafka                       | [2023-10-01 14:41:21,914] INFO [ThrottledChannelReaper-ControllerMutation]: Starting (kafka.server.ClientQuotaManager$ThrottledChannelReaper)
kafka                       | [2023-10-01 14:41:21,957] INFO Loading logs from log dirs ArraySeq(/var/lib/kafka/data) (kafka.log.LogManager)
kafka                       | [2023-10-01 14:41:21,961] INFO Attempting recovery for all logs in /var/lib/kafka/data since no clean shutdown file was found (kafka.log.LogManager)
kafka                       | [2023-10-01 14:41:21,987] INFO Loaded 0 logs in 28ms. (kafka.log.LogManager)
kafka                       | [2023-10-01 14:41:21,988] INFO Starting log cleanup with a period of 300000 ms. (kafka.log.LogManager)
kafka                       | [2023-10-01 14:41:21,993] INFO Starting log flusher with a default period of 9223372036854775807 ms. (kafka.log.LogManager)
kafka                       | [2023-10-01 14:41:22,009] INFO Starting the log cleaner (kafka.log.LogCleaner)
kafka                       | [2023-10-01 14:41:22,058] INFO [kafka-log-cleaner-thread-0]: Starting (kafka.log.LogCleaner)
kafka                       | [2023-10-01 14:41:22,088] INFO [feature-zk-node-event-process-thread]: Starting (kafka.server.FinalizedFeatureChangeListener$ChangeNotificationProcessorThread)
kafka                       | [2023-10-01 14:41:22,101] INFO Feature ZK node at path: /feature does not exist (kafka.server.FinalizedFeatureChangeListener)
kafka                       | [2023-10-01 14:41:22,129] INFO [BrokerToControllerChannelManager broker=1001 name=forwarding]: Starting (kafka.server.BrokerToControllerRequestThread)
kafka                       | [2023-10-01 14:41:22,593] INFO Updated connection-accept-rate max connection creation rate to 2147483647 (kafka.network.ConnectionQuotas)
kafka                       | [2023-10-01 14:41:22,597] INFO Awaiting socket connections on 0.0.0.0:29092. (kafka.network.DataPlaneAcceptor)
kafka                       | [2023-10-01 14:41:22,643] INFO [SocketServer listenerType=ZK_BROKER, nodeId=1001] Created data-plane acceptor and processors for endpoint : ListenerName(PLAINTEXT) (kafka.network.SocketServer)
kafka                       | [2023-10-01 14:41:22,643] INFO Updated connection-accept-rate max connection creation rate to 2147483647 (kafka.network.ConnectionQuotas)
kafka                       | [2023-10-01 14:41:22,644] INFO Awaiting socket connections on 0.0.0.0:9092. (kafka.network.DataPlaneAcceptor)
kafka                       | [2023-10-01 14:41:22,659] INFO [SocketServer listenerType=ZK_BROKER, nodeId=1001] Created data-plane acceptor and processors for endpoint : ListenerName(PLAINTEXT_HOST) (kafka.network.SocketServer)
kafka                       | [2023-10-01 14:41:22,679] INFO [BrokerToControllerChannelManager broker=1001 name=alterPartition]: Starting (kafka.server.BrokerToControllerRequestThread)
kafka                       | [2023-10-01 14:41:22,695] INFO [ExpirationReaper-1001-Produce]: Starting (kafka.server.DelayedOperationPurgatory$ExpiredOperationReaper)
kafka                       | [2023-10-01 14:41:22,695] INFO [ExpirationReaper-1001-Fetch]: Starting (kafka.server.DelayedOperationPurgatory$ExpiredOperationReaper)
kafka                       | [2023-10-01 14:41:22,699] INFO [ExpirationReaper-1001-DeleteRecords]: Starting (kafka.server.DelayedOperationPurgatory$ExpiredOperationReaper)
kafka                       | [2023-10-01 14:41:22,700] INFO [ExpirationReaper-1001-ElectLeader]: Starting (kafka.server.DelayedOperationPurgatory$ExpiredOperationReaper)
kafka                       | [2023-10-01 14:41:22,745] INFO [LogDirFailureHandler]: Starting (kafka.server.ReplicaManager$LogDirFailureHandler)
kafka                       | [2023-10-01 14:41:22,756] INFO Creating /brokers/ids/1001 (is it secure? false) (kafka.zk.KafkaZkClient)
kafka                       | [2023-10-01 14:41:22,777] INFO Stat of the created znode at /brokers/ids/1001 is: 28,28,1696171282769,1696171282769,1,0,0,72057606318718977,259,0,28
kafka                       |  (kafka.zk.KafkaZkClient)
kafka                       | [2023-10-01 14:41:22,780] INFO Registered broker 1001 at path /brokers/ids/1001 with addresses: PLAINTEXT://kafka:29092,PLAINTEXT_HOST://localhost:9092, czxid (broker epoch): 28 (kafka.zk.KafkaZkClient)
kafka                       | [2023-10-01 14:41:22,849] INFO [ControllerEventThread controllerId=1001] Starting (kafka.controller.ControllerEventManager$ControllerEventThread)
kafka                       | [2023-10-01 14:41:22,873] INFO [ExpirationReaper-1001-topic]: Starting (kafka.server.DelayedOperationPurgatory$ExpiredOperationReaper)
kafka                       | [2023-10-01 14:41:22,874] INFO [ExpirationReaper-1001-Heartbeat]: Starting (kafka.server.DelayedOperationPurgatory$ExpiredOperationReaper)
kafka                       | [2023-10-01 14:41:22,880] INFO [ExpirationReaper-1001-Rebalance]: Starting (kafka.server.DelayedOperationPurgatory$ExpiredOperationReaper)
kafka                       | [2023-10-01 14:41:22,891] INFO Successfully created /controller_epoch with initial epoch 0 (kafka.zk.KafkaZkClient)
kafka                       | [2023-10-01 14:41:22,900] INFO [GroupCoordinator 1001]: Starting up. (kafka.coordinator.group.GroupCoordinator)
kafka                       | [2023-10-01 14:41:22,904] INFO [Controller id=1001] 1001 successfully elected as the controller. Epoch incremented to 1 and epoch zk version is now 1 (kafka.controller.KafkaController)
kafka                       | [2023-10-01 14:41:22,910] INFO [GroupCoordinator 1001]: Startup complete. (kafka.coordinator.group.GroupCoordinator)
kafka                       | [2023-10-01 14:41:22,910] INFO [Controller id=1001] Creating FeatureZNode at path: /feature with contents: FeatureZNode(2,Enabled,Map()) (kafka.controller.KafkaController)
kafka                       | [2023-10-01 14:41:22,917] INFO Feature ZK node created at path: /feature (kafka.server.FinalizedFeatureChangeListener)
kafka                       | [2023-10-01 14:41:22,936] INFO [TransactionCoordinator id=1001] Starting up. (kafka.coordinator.transaction.TransactionCoordinator)
kafka                       | [2023-10-01 14:41:22,945] INFO [TransactionCoordinator id=1001] Startup complete. (kafka.coordinator.transaction.TransactionCoordinator)
kafka                       | [2023-10-01 14:41:22,954] INFO [Transaction Marker Channel Manager 1001]: Starting (kafka.coordinator.transaction.TransactionMarkerChannelManager)
kafka                       | [2023-10-01 14:41:22,969] INFO [Controller id=1001] Registering handlers (kafka.controller.KafkaController)
kafka                       | [2023-10-01 14:41:22,986] INFO [Controller id=1001] Deleting log dir event notifications (kafka.controller.KafkaController)
kafka                       | [2023-10-01 14:41:22,990] INFO [Controller id=1001] Deleting isr change notifications (kafka.controller.KafkaController)
kafka                       | [2023-10-01 14:41:22,994] INFO [Controller id=1001] Initializing controller context (kafka.controller.KafkaController)
kafka                       | [2023-10-01 14:41:23,005] INFO [MetadataCache brokerId=1001] Updated cache from existing <empty> to latest FinalizedFeaturesAndEpoch(features=Map(), epoch=0). (kafka.server.metadata.ZkMetadataCache)
kafka                       | [2023-10-01 14:41:23,021] INFO [Controller id=1001] Initialized broker epochs cache: HashMap(1001 -> 28) (kafka.controller.KafkaController)
kafka                       | [2023-10-01 14:41:23,024] INFO [ExpirationReaper--1-AlterAcls]: Starting (kafka.server.DelayedOperationPurgatory$ExpiredOperationReaper)
kafka                       | [2023-10-01 14:41:23,040] DEBUG [Controller id=1001] Register BrokerModifications handler for Set(1001) (kafka.controller.KafkaController)
kafka                       | [2023-10-01 14:41:23,047] DEBUG [Channel manager on controller 1001]: Controller 1001 trying to connect to broker 1001 (kafka.controller.ControllerChannelManager)
kafka                       | [2023-10-01 14:41:23,051] INFO [/config/changes-event-process-thread]: Starting (kafka.common.ZkNodeChangeNotificationListener$ChangeEventProcessThread)
kafka                       | [2023-10-01 14:41:23,063] INFO [SocketServer listenerType=ZK_BROKER, nodeId=1001] Enabling request processing. (kafka.network.SocketServer)
kafka                       | [2023-10-01 14:41:23,112] INFO Kafka version: 7.3.5-ccs (org.apache.kafka.common.utils.AppInfoParser)
kafka                       | [2023-10-01 14:41:23,112] INFO Kafka commitId: 96c5e882465d4b7e114ca6eae99d053ee5473929 (org.apache.kafka.common.utils.AppInfoParser)
kafka                       | [2023-10-01 14:41:23,112] INFO Kafka startTimeMs: 1696171283090 (org.apache.kafka.common.utils.AppInfoParser)
kafka                       | [2023-10-01 14:41:23,115] INFO [Controller id=1001] Currently active brokers in the cluster: Set(1001) (kafka.controller.KafkaController)
kafka                       | [2023-10-01 14:41:23,116] INFO [Controller id=1001] Currently shutting brokers in the cluster: HashSet() (kafka.controller.KafkaController)
kafka                       | [2023-10-01 14:41:23,116] INFO [Controller id=1001] Current list of topics in the cluster: HashSet() (kafka.controller.KafkaController)
kafka                       | [2023-10-01 14:41:23,116] INFO [Controller id=1001] Fetching topic deletions in progress (kafka.controller.KafkaController)
kafka                       | [2023-10-01 14:41:23,119] INFO [Controller id=1001] List of topics to be deleted:  (kafka.controller.KafkaController)
kafka                       | [2023-10-01 14:41:23,120] INFO [Controller id=1001] List of topics ineligible for deletion:  (kafka.controller.KafkaController)
kafka                       | [2023-10-01 14:41:23,120] INFO [Controller id=1001] Initializing topic deletion manager (kafka.controller.KafkaController)
kafka                       | [2023-10-01 14:41:23,121] INFO [Topic Deletion Manager 1001] Initializing manager with initial deletions: Set(), initial ineligible deletions: HashSet() (kafka.controller.TopicDeletionManager)
kafka                       | [2023-10-01 14:41:23,122] INFO [Controller id=1001] Sending update metadata request (kafka.controller.KafkaController)
kafka                       | [2023-10-01 14:41:23,122] INFO [KafkaServer id=1001] started (kafka.server.KafkaServer)
kafka                       | [2023-10-01 14:41:23,122] INFO [RequestSendThread controllerId=1001] Starting (kafka.controller.RequestSendThread)
kafka                       | [2023-10-01 14:41:23,130] INFO [Controller id=1001 epoch=1] Sending UpdateMetadata request to brokers HashSet(1001) for 0 partitions (state.change.logger)
kafka                       | [2023-10-01 14:41:23,147] INFO [ReplicaStateMachine controllerId=1001] Initializing replica state (kafka.controller.ZkReplicaStateMachine)
kafka                       | [2023-10-01 14:41:23,175] INFO [RequestSendThread controllerId=1001] Controller 1001 connected to kafka:29092 (id: 1001 rack: null) for sending state change requests (kafka.controller.RequestSendThread)
kafka                       | [2023-10-01 14:41:23,177] INFO [ReplicaStateMachine controllerId=1001] Triggering online replica state changes (kafka.controller.ZkReplicaStateMachine)
kafka                       | [2023-10-01 14:41:23,180] INFO [ReplicaStateMachine controllerId=1001] Triggering offline replica state changes (kafka.controller.ZkReplicaStateMachine)
kafka                       | [2023-10-01 14:41:23,191] DEBUG [ReplicaStateMachine controllerId=1001] Started replica state machine with initial state -> HashMap() (kafka.controller.ZkReplicaStateMachine)
kafka                       | [2023-10-01 14:41:23,192] INFO [PartitionStateMachine controllerId=1001] Initializing partition state (kafka.controller.ZkPartitionStateMachine)
kafka                       | [2023-10-01 14:41:23,194] INFO [PartitionStateMachine controllerId=1001] Triggering online partition state changes (kafka.controller.ZkPartitionStateMachine)
kafka                       | [2023-10-01 14:41:23,220] DEBUG [PartitionStateMachine controllerId=1001] Started partition state machine with initial state -> HashMap() (kafka.controller.ZkPartitionStateMachine)
kafka                       | [2023-10-01 14:41:23,226] INFO [Controller id=1001] Ready to serve as the new controller with epoch 1 (kafka.controller.KafkaController)
kafka                       | [2023-10-01 14:41:23,253] INFO [Controller id=1001] Partitions undergoing preferred replica election:  (kafka.controller.KafkaController)
kafka                       | [2023-10-01 14:41:23,257] INFO [Controller id=1001] Partitions that completed preferred replica election:  (kafka.controller.KafkaController)
kafka                       | [2023-10-01 14:41:23,257] INFO [Controller id=1001] Skipping preferred replica election for partitions due to topic deletion:  (kafka.controller.KafkaController)
kafka                       | [2023-10-01 14:41:23,257] INFO [Controller id=1001] Resuming preferred replica election for partitions:  (kafka.controller.KafkaController)
kafka                       | [2023-10-01 14:41:23,262] INFO [Controller id=1001] Starting replica leader election (PREFERRED) for partitions  triggered by ZkTriggered (kafka.controller.KafkaController)
kafka                       | [2023-10-01 14:41:23,283] INFO [Controller id=1001] Starting the controller scheduler (kafka.controller.KafkaController)
kafka                       | [2023-10-01 14:41:23,333] TRACE [Controller id=1001 epoch=1] Received response UpdateMetadataResponseData(errorCode=0) for request UPDATE_METADATA with correlation id 0 sent to broker kafka:29092 (id: 1001 rack: null) (state.change.logger)
kafka                       | [2023-10-01 14:41:23,365] INFO [BrokerToControllerChannelManager broker=1001 name=forwarding]: Recorded new controller, from now on will use node kafka:29092 (id: 1001 rack: null) (kafka.server.BrokerToControllerRequestThread)
kafka                       | [2023-10-01 14:41:23,404] INFO [BrokerToControllerChannelManager broker=1001 name=alterPartition]: Recorded new controller, from now on will use node kafka:29092 (id: 1001 rack: null) (kafka.server.BrokerToControllerRequestThread)
kafka                       | [2023-10-01 14:41:28,285] INFO [Controller id=1001] Processing automatic preferred replica leader election (kafka.controller.KafkaController)
kafka                       | [2023-10-01 14:41:28,286] TRACE [Controller id=1001] Checking need to trigger auto leader balancing (kafka.controller.KafkaController)
fashion-mnist-classifier    | Running tests...
fashion-mnist-classifier    | 2023-10-01 14:41:30,815 — __main__ — INFO — Testing datasets
fashion-mnist-classifier    | 2023-10-01 14:41:30,863 — __main__ — INFO — Datasets collected
fashion-mnist-classifier    | 2023-10-01 14:41:30,863 — __main__ — INFO — Testing datasets len...
fashion-mnist-classifier    | 2023-10-01 14:41:30,863 — __main__ — INFO — Testing datasets len passed!
fashion-mnist-classifier    | .2023-10-01 14:41:30,864 — __main__ — INFO — Testing datasets
fashion-mnist-classifier    | 2023-10-01 14:41:30,911 — __main__ — INFO — Datasets collected
fashion-mnist-classifier    | 2023-10-01 14:41:30,911 — __main__ — INFO — Testing datasets types...
fashion-mnist-classifier    | 2023-10-01 14:41:30,911 — __main__ — INFO — Testing datasets types passed!
fashion-mnist-classifier    | .
fashion-mnist-classifier    | ----------------------------------------------------------------------
fashion-mnist-classifier    | Ran 2 tests in 0.097s
fashion-mnist-classifier    | 
fashion-mnist-classifier    | OK
kafka                       | [2023-10-01 14:41:35,206] INFO Creating topic kafka-ckpt with configuration {} and initial partition assignment HashMap(0 -> ArrayBuffer(1001)) (kafka.zk.AdminZkClient)
kafka                       | [2023-10-01 14:41:35,243] INFO [Controller id=1001] New topics: [Set(kafka-ckpt)], deleted topics: [HashSet()], new partition replica assignment [Set(TopicIdReplicaAssignment(kafka-ckpt,Some(JEBHuSVSTC-4b0wSYaiylg),Map(kafka-ckpt-0 -> ReplicaAssignment(replicas=1001, addingReplicas=, removingReplicas=))))] (kafka.controller.KafkaController)
kafka                       | [2023-10-01 14:41:35,244] INFO [Controller id=1001] New partition creation callback for kafka-ckpt-0 (kafka.controller.KafkaController)
kafka                       | [2023-10-01 14:41:35,248] INFO [Controller id=1001 epoch=1] Changed partition kafka-ckpt-0 state from NonExistentPartition to NewPartition with assigned replicas 1001 (state.change.logger)
kafka                       | [2023-10-01 14:41:35,246] INFO Creating topic kafka-metrics with configuration {} and initial partition assignment HashMap(0 -> ArrayBuffer(1001)) (kafka.zk.AdminZkClient)
kafka                       | [2023-10-01 14:41:35,251] INFO [Controller id=1001 epoch=1] Sending UpdateMetadata request to brokers HashSet() for 0 partitions (state.change.logger)
kafka                       | [2023-10-01 14:41:35,267] TRACE [Controller id=1001 epoch=1] Changed state of replica 1001 for partition kafka-ckpt-0 from NonExistentReplica to NewReplica (state.change.logger)
kafka                       | [2023-10-01 14:41:35,268] INFO [Controller id=1001 epoch=1] Sending UpdateMetadata request to brokers HashSet() for 0 partitions (state.change.logger)
kafka                       | [2023-10-01 14:41:35,289] INFO [Controller id=1001 epoch=1] Changed partition kafka-ckpt-0 from NewPartition to OnlinePartition with state LeaderAndIsr(leader=1001, leaderEpoch=0, isr=List(1001), leaderRecoveryState=RECOVERED, partitionEpoch=0) (state.change.logger)
kafka                       | [2023-10-01 14:41:35,292] TRACE [Controller id=1001 epoch=1] Sending become-leader LeaderAndIsr request LeaderAndIsrPartitionState(topicName='kafka-ckpt', partitionIndex=0, controllerEpoch=1, leader=1001, leaderEpoch=0, isr=[1001], partitionEpoch=0, replicas=[1001], addingReplicas=[], removingReplicas=[], isNew=true, leaderRecoveryState=0) to broker 1001 for partition kafka-ckpt-0 (state.change.logger)
kafka                       | [2023-10-01 14:41:35,294] INFO [Controller id=1001 epoch=1] Sending LeaderAndIsr request to broker 1001 with 1 become-leader and 0 become-follower partitions (state.change.logger)
kafka                       | [2023-10-01 14:41:35,297] INFO [Controller id=1001 epoch=1] Sending UpdateMetadata request to brokers HashSet(1001) for 1 partitions (state.change.logger)
kafka                       | [2023-10-01 14:41:35,298] TRACE [Controller id=1001 epoch=1] Changed state of replica 1001 for partition kafka-ckpt-0 from NewReplica to OnlineReplica (state.change.logger)
kafka                       | [2023-10-01 14:41:35,299] INFO [Controller id=1001 epoch=1] Sending UpdateMetadata request to brokers HashSet() for 0 partitions (state.change.logger)
kafka                       | [2023-10-01 14:41:35,303] INFO [Controller id=1001] New topics: [Set(kafka-metrics)], deleted topics: [HashSet()], new partition replica assignment [Set(TopicIdReplicaAssignment(kafka-metrics,Some(Gnoq9MFuS1uoL-k08dm_Zw),Map(kafka-metrics-0 -> ReplicaAssignment(replicas=1001, addingReplicas=, removingReplicas=))))] (kafka.controller.KafkaController)
kafka                       | [2023-10-01 14:41:35,303] INFO [Controller id=1001] New partition creation callback for kafka-metrics-0 (kafka.controller.KafkaController)
kafka                       | [2023-10-01 14:41:35,304] INFO [Controller id=1001 epoch=1] Changed partition kafka-metrics-0 state from NonExistentPartition to NewPartition with assigned replicas 1001 (state.change.logger)
kafka                       | [2023-10-01 14:41:35,304] INFO [Controller id=1001 epoch=1] Sending UpdateMetadata request to brokers HashSet() for 0 partitions (state.change.logger)
kafka                       | [2023-10-01 14:41:35,305] TRACE [Controller id=1001 epoch=1] Changed state of replica 1001 for partition kafka-metrics-0 from NonExistentReplica to NewReplica (state.change.logger)
kafka                       | [2023-10-01 14:41:35,305] INFO [Controller id=1001 epoch=1] Sending UpdateMetadata request to brokers HashSet() for 0 partitions (state.change.logger)
kafka                       | [2023-10-01 14:41:35,307] INFO [Broker id=1001] Handling LeaderAndIsr request correlationId 1 from controller 1001 for 1 partitions (state.change.logger)
kafka                       | [2023-10-01 14:41:35,309] TRACE [Broker id=1001] Received LeaderAndIsr request LeaderAndIsrPartitionState(topicName='kafka-ckpt', partitionIndex=0, controllerEpoch=1, leader=1001, leaderEpoch=0, isr=[1001], partitionEpoch=0, replicas=[1001], addingReplicas=[], removingReplicas=[], isNew=true, leaderRecoveryState=0) correlation id 1 from controller 1001 epoch 1 (state.change.logger)
kafka                       | [2023-10-01 14:41:35,317] INFO [Controller id=1001 epoch=1] Changed partition kafka-metrics-0 from NewPartition to OnlinePartition with state LeaderAndIsr(leader=1001, leaderEpoch=0, isr=List(1001), leaderRecoveryState=RECOVERED, partitionEpoch=0) (state.change.logger)
kafka                       | [2023-10-01 14:41:35,318] TRACE [Controller id=1001 epoch=1] Sending become-leader LeaderAndIsr request LeaderAndIsrPartitionState(topicName='kafka-metrics', partitionIndex=0, controllerEpoch=1, leader=1001, leaderEpoch=0, isr=[1001], partitionEpoch=0, replicas=[1001], addingReplicas=[], removingReplicas=[], isNew=true, leaderRecoveryState=0) to broker 1001 for partition kafka-metrics-0 (state.change.logger)
kafka                       | [2023-10-01 14:41:35,319] INFO [Controller id=1001 epoch=1] Sending LeaderAndIsr request to broker 1001 with 1 become-leader and 0 become-follower partitions (state.change.logger)
kafka                       | [2023-10-01 14:41:35,322] INFO [Controller id=1001 epoch=1] Sending UpdateMetadata request to brokers HashSet(1001) for 1 partitions (state.change.logger)
kafka                       | [2023-10-01 14:41:35,324] TRACE [Controller id=1001 epoch=1] Changed state of replica 1001 for partition kafka-metrics-0 from NewReplica to OnlineReplica (state.change.logger)
kafka                       | [2023-10-01 14:41:35,324] INFO [Controller id=1001 epoch=1] Sending UpdateMetadata request to brokers HashSet() for 0 partitions (state.change.logger)
kafka                       | [2023-10-01 14:41:35,339] TRACE [Broker id=1001] Handling LeaderAndIsr request correlationId 1 from controller 1001 epoch 1 starting the become-leader transition for partition kafka-ckpt-0 (state.change.logger)
kafka                       | [2023-10-01 14:41:35,340] INFO [ReplicaFetcherManager on broker 1001] Removed fetcher for partitions Set(kafka-ckpt-0) (kafka.server.ReplicaFetcherManager)
kafka                       | [2023-10-01 14:41:35,341] INFO [Broker id=1001] Stopped fetchers as part of LeaderAndIsr request correlationId 1 from controller 1001 epoch 1 as part of the become-leader transition for 1 partitions (state.change.logger)
kafka                       | [2023-10-01 14:41:35,401] INFO [LogLoader partition=kafka-ckpt-0, dir=/var/lib/kafka/data] Loading producer state till offset 0 with message format version 2 (kafka.log.UnifiedLog$)
kafka                       | [2023-10-01 14:41:35,415] INFO Created log for partition kafka-ckpt-0 in /var/lib/kafka/data/kafka-ckpt-0 with properties {} (kafka.log.LogManager)
kafka                       | [2023-10-01 14:41:35,417] INFO [Partition kafka-ckpt-0 broker=1001] No checkpointed highwatermark is found for partition kafka-ckpt-0 (kafka.cluster.Partition)
kafka                       | [2023-10-01 14:41:35,418] INFO [Partition kafka-ckpt-0 broker=1001] Log loaded for partition kafka-ckpt-0 with initial high watermark 0 (kafka.cluster.Partition)
kafka                       | [2023-10-01 14:41:35,420] INFO [Broker id=1001] Leader kafka-ckpt-0 with topic id Some(JEBHuSVSTC-4b0wSYaiylg) starts at leader epoch 0 from offset 0 with partition epoch 0, high watermark 0, ISR [1001], adding replicas [] and removing replicas []. Previous leader epoch was -1. (state.change.logger)
kafka                       | [2023-10-01 14:41:35,437] TRACE [Broker id=1001] Completed LeaderAndIsr request correlationId 1 from controller 1001 epoch 1 for the become-leader transition for partition kafka-ckpt-0 (state.change.logger)
kafka                       | [2023-10-01 14:41:35,444] INFO [Broker id=1001] Finished LeaderAndIsr request in 142ms correlationId 1 from controller 1001 for 1 partitions (state.change.logger)
kafka                       | [2023-10-01 14:41:35,448] TRACE [Controller id=1001 epoch=1] Received response LeaderAndIsrResponseData(errorCode=0, partitionErrors=[], topics=[LeaderAndIsrTopicError(topicId=JEBHuSVSTC-4b0wSYaiylg, partitionErrors=[LeaderAndIsrPartitionError(topicName='', partitionIndex=0, errorCode=0)])]) for request LEADER_AND_ISR with correlation id 1 sent to broker kafka:29092 (id: 1001 rack: null) (state.change.logger)
kafka                       | [2023-10-01 14:41:35,455] TRACE [Broker id=1001] Cached leader info UpdateMetadataPartitionState(topicName='kafka-ckpt', partitionIndex=0, controllerEpoch=1, leader=1001, leaderEpoch=0, isr=[1001], zkVersion=0, replicas=[1001], offlineReplicas=[]) for partition kafka-ckpt-0 in response to UpdateMetadata request sent by controller 1001 epoch 1 with correlation id 2 (state.change.logger)
kafka                       | [2023-10-01 14:41:35,457] INFO [Broker id=1001] Add 1 partitions and deleted 0 partitions from metadata cache in response to UpdateMetadata request sent by controller 1001 epoch 1 with correlation id 2 (state.change.logger)
kafka                       | [2023-10-01 14:41:35,459] TRACE [Controller id=1001 epoch=1] Received response UpdateMetadataResponseData(errorCode=0) for request UPDATE_METADATA with correlation id 2 sent to broker kafka:29092 (id: 1001 rack: null) (state.change.logger)
kafka                       | [2023-10-01 14:41:35,460] INFO [Broker id=1001] Handling LeaderAndIsr request correlationId 3 from controller 1001 for 1 partitions (state.change.logger)
kafka                       | [2023-10-01 14:41:35,461] TRACE [Broker id=1001] Received LeaderAndIsr request LeaderAndIsrPartitionState(topicName='kafka-metrics', partitionIndex=0, controllerEpoch=1, leader=1001, leaderEpoch=0, isr=[1001], partitionEpoch=0, replicas=[1001], addingReplicas=[], removingReplicas=[], isNew=true, leaderRecoveryState=0) correlation id 3 from controller 1001 epoch 1 (state.change.logger)
kafka                       | [2023-10-01 14:41:35,463] TRACE [Broker id=1001] Handling LeaderAndIsr request correlationId 3 from controller 1001 epoch 1 starting the become-leader transition for partition kafka-metrics-0 (state.change.logger)
kafka                       | [2023-10-01 14:41:35,464] INFO [ReplicaFetcherManager on broker 1001] Removed fetcher for partitions Set(kafka-metrics-0) (kafka.server.ReplicaFetcherManager)
kafka                       | [2023-10-01 14:41:35,464] INFO [Broker id=1001] Stopped fetchers as part of LeaderAndIsr request correlationId 3 from controller 1001 epoch 1 as part of the become-leader transition for 1 partitions (state.change.logger)
kafka                       | [2023-10-01 14:41:35,469] INFO [LogLoader partition=kafka-metrics-0, dir=/var/lib/kafka/data] Loading producer state till offset 0 with message format version 2 (kafka.log.UnifiedLog$)
kafka                       | [2023-10-01 14:41:35,471] INFO Created log for partition kafka-metrics-0 in /var/lib/kafka/data/kafka-metrics-0 with properties {} (kafka.log.LogManager)
kafka                       | [2023-10-01 14:41:35,472] INFO [Partition kafka-metrics-0 broker=1001] No checkpointed highwatermark is found for partition kafka-metrics-0 (kafka.cluster.Partition)
kafka                       | [2023-10-01 14:41:35,473] INFO [Partition kafka-metrics-0 broker=1001] Log loaded for partition kafka-metrics-0 with initial high watermark 0 (kafka.cluster.Partition)
kafka                       | [2023-10-01 14:41:35,474] INFO [Broker id=1001] Leader kafka-metrics-0 with topic id Some(Gnoq9MFuS1uoL-k08dm_Zw) starts at leader epoch 0 from offset 0 with partition epoch 0, high watermark 0, ISR [1001], adding replicas [] and removing replicas []. Previous leader epoch was -1. (state.change.logger)
kafka                       | [2023-10-01 14:41:35,477] TRACE [Broker id=1001] Completed LeaderAndIsr request correlationId 3 from controller 1001 epoch 1 for the become-leader transition for partition kafka-metrics-0 (state.change.logger)
kafka                       | [2023-10-01 14:41:35,478] INFO [Broker id=1001] Finished LeaderAndIsr request in 18ms correlationId 3 from controller 1001 for 1 partitions (state.change.logger)
kafka                       | [2023-10-01 14:41:35,479] TRACE [Controller id=1001 epoch=1] Received response LeaderAndIsrResponseData(errorCode=0, partitionErrors=[], topics=[LeaderAndIsrTopicError(topicId=Gnoq9MFuS1uoL-k08dm_Zw, partitionErrors=[LeaderAndIsrPartitionError(topicName='', partitionIndex=0, errorCode=0)])]) for request LEADER_AND_ISR with correlation id 3 sent to broker kafka:29092 (id: 1001 rack: null) (state.change.logger)
kafka                       | [2023-10-01 14:41:35,482] TRACE [Broker id=1001] Cached leader info UpdateMetadataPartitionState(topicName='kafka-metrics', partitionIndex=0, controllerEpoch=1, leader=1001, leaderEpoch=0, isr=[1001], zkVersion=0, replicas=[1001], offlineReplicas=[]) for partition kafka-metrics-0 in response to UpdateMetadata request sent by controller 1001 epoch 1 with correlation id 4 (state.change.logger)
kafka                       | [2023-10-01 14:41:35,482] INFO [Broker id=1001] Add 1 partitions and deleted 0 partitions from metadata cache in response to UpdateMetadata request sent by controller 1001 epoch 1 with correlation id 4 (state.change.logger)
kafka                       | [2023-10-01 14:41:35,491] TRACE [Controller id=1001 epoch=1] Received response UpdateMetadataResponseData(errorCode=0) for request UPDATE_METADATA with correlation id 4 sent to broker kafka:29092 (id: 1001 rack: null) (state.change.logger)
fashion-mnist-classifier    | 2023-10-01 14:41:35,813 — db_utils — INFO — Using ansible to get DB credentials
fashion-mnist-classifier    | 2023-10-01 14:41:35,854 — __main__ — INFO — Producer just sent msg with CKPT topic
fashion-mnist-classifier    | 2023-10-01 14:41:36,003 — __main__ — INFO — CKPT Consumer got msg: ConsumerRecord(topic='kafka-ckpt', partition=0, offset=0, timestamp=1696171295853, timestamp_type=0, key=None, value='checkpoints/CNNModel_FashionMNIST_best_metric_model.pth', headers=[], checksum=None, serialized_key_size=-1, serialized_value_size=55, serialized_header_size=-1)
fashion-mnist-classifier    | 2023-10-01 14:41:36,006 — __main__ — INFO — Collecting dataloaders
fashion-mnist-classifier    | 2023-10-01 14:41:36,069 — __main__ — INFO — Loading classifier
fashion-mnist-classifier    | 2023-10-01 14:41:36,086 — __main__ — INFO — Checkpoint loaded
fashion-mnist-classifier    | 2023-10-01 14:41:36,087 — __main__ — INFO — Testing classifier metrics...
fashion-mnist-classifier    | 2023-10-01 14:41:36,087 — classifier — INFO — *************************
fashion-mnist-classifier    | 2023-10-01 14:41:36,087 — classifier — INFO — >> Testing CNNModel network
fashion-mnist-classifier    | 
  0%|          | 0/40 [00:00<?, ?it/s]
  5%|▌         | 2/40 [00:00<00:02, 16.62it/s]
 10%|█         | 4/40 [00:00<00:02, 17.51it/s]
 15%|█▌        | 6/40 [00:00<00:01, 18.12it/s]
 20%|██        | 8/40 [00:00<00:01, 18.37it/s]
 25%|██▌       | 10/40 [00:00<00:01, 18.46it/s]
 30%|███       | 12/40 [00:00<00:01, 18.79it/s]
 38%|███▊      | 15/40 [00:00<00:01, 20.01it/s]
 45%|████▌     | 18/40 [00:00<00:01, 20.42it/s]
 52%|█████▎    | 21/40 [00:01<00:00, 20.56it/s]
 60%|██████    | 24/40 [00:01<00:00, 20.89it/s]
 68%|██████▊   | 27/40 [00:01<00:00, 21.09it/s]
 75%|███████▌  | 30/40 [00:01<00:00, 16.82it/s]
 80%|████████  | 32/40 [00:01<00:00, 16.97it/s]
 85%|████████▌ | 34/40 [00:01<00:00, 17.58it/s]
 92%|█████████▎| 37/40 [00:01<00:00, 18.80it/s]
 98%|█████████▊| 39/40 [00:02<00:00, 18.69it/s]
100%|██████████| 40/40 [00:02<00:00, 19.22it/s]
fashion-mnist-classifier    | 2023-10-01 14:41:39,128 — classifier — INFO — Total test loss: 0.5603217876434327
fashion-mnist-classifier    | 2023-10-01 14:41:39,128 — classifier — INFO — Total test accuracy: 0.7992
fashion-mnist-classifier    | 2023-10-01 14:41:39,128 — classifier — INFO — Total test F1_macro score: 0.7955895683253991
fashion-mnist-classifier    | 2023-10-01 14:41:39,128 — classifier — INFO — Confusion matrix:
fashion-mnist-classifier    | 2023-10-01 14:41:39,128 — classifier — INFO — [[806   2   9  82   7   9  66   0  19   0]
fashion-mnist-classifier    |  [  7 919  15  48   7   0   2   0   2   0]
fashion-mnist-classifier    |  [ 23   0 651  11 171   2 126   0  16   0]
fashion-mnist-classifier    |  [ 31   9   2 856  26   2  69   0   5   0]
fashion-mnist-classifier    |  [  1   1 122  51 707   1 109   0   8   0]
fashion-mnist-classifier    |  [  0   0   0   2   0 924   0  53   2  19]
fashion-mnist-classifier    |  [232   1 124  56 153   3 398   0  33   0]
fashion-mnist-classifier    |  [  0   0   0   0   0  67   0 863   1  69]
fashion-mnist-classifier    |  [  4   1   9   9   1   9  13   7 946   1]
fashion-mnist-classifier    |  [  0   0   0   2   0  26   0  49   1 922]]
fashion-mnist-classifier    | 2023-10-01 14:41:39,134 — __main__ — INFO — Producer just sent msg with METRICS topic
fashion-mnist-classifier    | .2023-10-01 14:41:39,135 — __main__ — INFO — Tests passed!
fashion-mnist-classifier    | 
fashion-mnist-classifier    | ----------------------------------------------------------------------2023-10-01 14:41:39,137 — __main__ — INFO — Running METRICS Consumer...
fashion-mnist-classifier    | 
fashion-mnist-classifier    | Ran 1 test in 3.131s
fashion-mnist-classifier    | 
fashion-mnist-classifier    | OK
fashion-mnist-classifier    | 2023-10-01 14:41:39,247 — __main__ — INFO — METRICS Consumer got msg: ConsumerRecord(topic='kafka-metrics', partition=0, offset=0, timestamp=1696171299133, timestamp_type=0, key=None, value='0.5603217876434327 0.7992 0.7955895683253991', headers=[], checksum=None, serialized_key_size=-1, serialized_value_size=44, serialized_header_size=-1)
fashion-mnist-classifier    | 2023-10-01 14:41:39,251 — __main__ — INFO — ==================================================
fashion-mnist-classifier    | 2023-10-01 14:41:39,251 — __main__ — INFO — Results table:
fashion-mnist-classifier    | 2023-10-01 14:41:39,251 — __main__ — INFO — ----------------------------------------------------------
fashion-mnist-classifier    |  id | epoch_loss         | epoch_acc | f1_macro           
fashion-mnist-classifier    | ----+--------------------+-----------+--------------------
fashion-mnist-classifier    |   1 | 0.5603217876434327 | 0.7992    | 0.7955895683253991 
fashion-mnist-classifier    | ----------------------------------------------------------
fashion-mnist-classifier    | (1 row)
fashion-mnist-classifier    | 
fashion-mnist-classifier    | 2023-10-01 14:41:39,252 — __main__ — INFO — --------------------
fashion-mnist-classifier    | 2023-10-01 14:41:39,253 — __main__ — INFO — Weights table:
fashion-mnist-classifier    | 2023-10-01 14:41:39,253 — __main__ — INFO — --------------------------------------------------------------
fashion-mnist-classifier    |  id | model_path                                              
fashion-mnist-classifier    | ----+---------------------------------------------------------
fashion-mnist-classifier    |   1 | checkpoints/CNNModel_FashionMNIST_best_metric_model.pth 
fashion-mnist-classifier    | --------------------------------------------------------------
fashion-mnist-classifier    | (1 row)
fashion-mnist-classifier    | 
fashion-mnist-classifier    | Name                                      Stmts   Miss  Cover   Missing
fashion-mnist-classifier    | -----------------------------------------------------------------------
fashion-mnist-classifier    | src/ansible_credential_utils.py               8      2    75%   11-12
fashion-mnist-classifier    | src/classifier.py                           105     40    62%   36, 56-58, 77, 80-126
fashion-mnist-classifier    | src/dataset_utils.py                         15      0   100%
fashion-mnist-classifier    | src/db_utils.py                              39      3    92%   34, 49-51
fashion-mnist-classifier    | src/fashion_mnist_classifier.py              19      0   100%
fashion-mnist-classifier    | src/kafka_utils.py                           30      0   100%
fashion-mnist-classifier    | src/logger.py                                26      0   100%
fashion-mnist-classifier    | src/model.py                                 10      0   100%
fashion-mnist-classifier    | src/unit_tests/test_training_results.py      63      1    98%   56
fashion-mnist-classifier    | -----------------------------------------------------------------------
fashion-mnist-classifier    | TOTAL                                       315     46    85%
fashion-mnist-classifier exited with code 0
Stopping kafka                    ... 
Stopping database                 ... 
Stopping zookeeper                ... 
Stopping database                 ... done
Stopping kafka                    ... done
Stopping zookeeper                ... done
Aborting on container exit...
```