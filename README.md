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

#### Дополнительная информация

* Пароли для авторизации на DockerHub и на удаленном сервере были сохранены с использованием GitHub Secrets
* Ссылка на загруженный Docker
  образ: https://hub.docker.com/repository/docker/izhanvarsky/bigdata-hw1-fashion-mnist-classifier
* Результаты тестирования можно найти
  здесь: https://github.com/IzhanVarsky/big-data-hw1/actions/workflows/docker_run_tests.yaml
* Пример результатов тестирования:

```
Creating network "big-data-hw_default" with the default driver
Creating database ... 
Creating database ... done
Creating big-data-hw_fashion-mnist-classifier_1 ... 
Creating big-data-hw_fashion-mnist-classifier_1 ... done
Attaching to database, big-data-hw_fashion-mnist-classifier_1
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
database                    | creating configuration files ... ok
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
database                    | waiting for server to start....2023-09-30 18:26:03.599 UTC [48] LOG:  starting PostgreSQL 16.0 (Debian 16.0-1.pgdg120+1) on x86_64-pc-linux-gnu, compiled by gcc (Debian 12.2.0-14) 12.2.0, 64-bit
database                    | 2023-09-30 18:26:03.600 UTC [48] LOG:  listening on Unix socket "/var/run/***ql/.s.PGSQL.5432"
database                    | 2023-09-30 18:26:03.604 UTC [51] LOG:  database system was shut down at 2023-09-30 18:26:03 UTC
database                    | 2023-09-30 18:26:03.609 UTC [48] LOG:  database system is ready to accept connections
database                    |  done
database                    | server started
database                    | 
database                    | /usr/local/bin/docker-entrypoint.sh: running /docker-entrypoint-initdb.d/init.sql
database                    | CREATE TABLE
database                    | INSERT 0 1
database                    | CREATE TABLE
database                    | 
database                    | 
database                    | 2023-09-30 18:26:03.785 UTC [48] LOG:  received fast shutdown request
database                    | waiting for server to shut down....2023-09-30 18:26:03.786 UTC [48] LOG:  aborting any active transactions
database                    | 2023-09-30 18:26:03.790 UTC [48] LOG:  background worker "logical replication launcher" (PID 54) exited with exit code 1
database                    | 2023-09-30 18:26:03.790 UTC [49] LOG:  shutting down
database                    | 2023-09-30 18:26:03.791 UTC [49] LOG:  checkpoint starting: shutdown immediate
database                    | 2023-09-30 18:26:03.798 UTC [49] LOG:  checkpoint complete: wrote 65 buffers (0.4%); 0 WAL file(s) added, 0 removed, 0 recycled; write=0.004 s, sync=0.002 s, total=0.008 s; sync files=48, longest=0.001 s, average=0.001 s; distance=207 kB, estimate=207 kB; lsn=0/151E718, redo lsn=0/151E718
database                    | 2023-09-30 18:26:03.802 UTC [48] LOG:  database system is shut down
database                    |  done
database                    | server stopped
database                    | 
database                    | PostgreSQL init process complete; ready for start up.
database                    | 
database                    | 2023-09-30 18:26:03.914 UTC [1] LOG:  starting PostgreSQL 16.0 (Debian 16.0-1.pgdg120+1) on x86_64-pc-linux-gnu, compiled by gcc (Debian 12.2.0-14) 12.2.0, 64-bit
database                    | 2023-09-30 18:26:03.915 UTC [1] LOG:  listening on IPv4 address "0.0.0.0", port 5432
database                    | 2023-09-30 18:26:03.915 UTC [1] LOG:  listening on IPv6 address "::", port 5432
database                    | 2023-09-30 18:26:03.918 UTC [1] LOG:  listening on Unix socket "/var/run/***ql/.s.PGSQL.5432"
database                    | 2023-09-30 18:26:03.921 UTC [64] LOG:  database system was shut down at 2023-09-30 18:26:03 UTC
database                    | 2023-09-30 18:26:03.927 UTC [1] LOG:  database system is ready to accept connections
fashion-mnist-classifier_1  | 2023-09-30 18:26:10,571 — __main__ — INFO — Testing datasets
fashion-mnist-classifier_1  | 2023-09-30 18:26:10,620 — __main__ — INFO — Datasets collected
fashion-mnist-classifier_1  | 2023-09-30 18:26:10,620 — __main__ — INFO — Testing datasets len...
fashion-mnist-classifier_1  | 2023-09-30 18:26:10,620 — __main__ — INFO — Testing datasets len passed!
fashion-mnist-classifier_1  | .2023-09-30 18:26:10,621 — __main__ — INFO — Testing datasets
fashion-mnist-classifier_1  | 2023-09-30 18:26:10,668 — __main__ — INFO — Datasets collected
fashion-mnist-classifier_1  | 2023-09-30 18:26:10,668 — __main__ — INFO — Testing datasets types...
fashion-mnist-classifier_1  | 2023-09-30 18:26:10,669 — __main__ — INFO — Testing datasets types passed!
fashion-mnist-classifier_1  | .
fashion-mnist-classifier_1  | ----------------------------------------------------------------------
fashion-mnist-classifier_1  | Ran 2 tests in 0.098s
fashion-mnist-classifier_1  | 
fashion-mnist-classifier_1  | OK
fashion-mnist-classifier_1  | 2023-09-30 18:26:14,633 — db_utils — INFO — Using ansible to get DB credentials
fashion-mnist-classifier_1  | 2023-09-30 18:26:14,676 — __main__ — INFO — Results table:
fashion-mnist-classifier_1  | 2023-09-30 18:26:14,677 — __main__ — INFO — --------------------------------------------------------------
fashion-mnist-classifier_1  |  id | model_path                                              
fashion-mnist-classifier_1  | ----+---------------------------------------------------------
fashion-mnist-classifier_1  |   1 | checkpoints/CNNModel_FashionMNIST_best_metric_model.pth 
fashion-mnist-classifier_1  | --------------------------------------------------------------
fashion-mnist-classifier_1  | (1 row)
fashion-mnist-classifier_1  | 
fashion-mnist-classifier_1  | 2023-09-30 18:26:14,681 — __main__ — INFO — Collecting dataloaders
fashion-mnist-classifier_1  | 2023-09-30 18:26:14,729 — __main__ — INFO — Loading classifier
fashion-mnist-classifier_1  | 2023-09-30 18:26:14,741 — __main__ — INFO — Checkpoint loaded
fashion-mnist-classifier_1  | 2023-09-30 18:26:14,742 — __main__ — INFO — Testing classifier metrics...
fashion-mnist-classifier_1  | 2023-09-30 18:26:14,742 — classifier — INFO — *************************
fashion-mnist-classifier_1  | 2023-09-30 18:26:14,742 — classifier — INFO — >> Testing CNNModel network
fashion-mnist-classifier_1  | 
  0%|          | 0/40 [00:00<?, ?it/s]
  5%|▌         | 2/40 [00:00<00:02, 17.34it/s]
 10%|█         | 4/40 [00:00<00:02, 15.44it/s]
 15%|█▌        | 6/40 [00:00<00:02, 16.71it/s]
 20%|██        | 8/40 [00:00<00:01, 17.59it/s]
 25%|██▌       | 10/40 [00:00<00:01, 18.05it/s]
 30%|███       | 12/40 [00:00<00:01, 18.18it/s]
 35%|███▌      | 14/40 [00:00<00:01, 18.46it/s]
 40%|████      | 16/40 [00:00<00:01, 18.60it/s]
 45%|████▌     | 18/40 [00:01<00:01, 18.32it/s]
 50%|█████     | 20/40 [00:01<00:01, 18.71it/s]
 55%|█████▌    | 22/40 [00:01<00:00, 18.95it/s]
 60%|██████    | 24/40 [00:01<00:00, 18.19it/s]
 65%|██████▌   | 26/40 [00:01<00:00, 18.09it/s]
 70%|███████   | 28/40 [00:01<00:00, 18.31it/s]
 75%|███████▌  | 30/40 [00:01<00:00, 18.71it/s]
 80%|████████  | 32/40 [00:01<00:00, 18.95it/s]
 85%|████████▌ | 34/40 [00:01<00:00, 19.24it/s]
 90%|█████████ | 36/40 [00:01<00:00, 19.38it/s]
 95%|█████████▌| 38/40 [00:02<00:00, 19.43it/s]
100%|██████████| 40/40 [00:02<00:00, 18.94it/s]
fashion-mnist-classifier_1  | 2023-09-30 18:26:17,982 — classifier — INFO — Total test loss: 0.5603217876434327
fashion-mnist-classifier_1  | 2023-09-30 18:26:17,982 — classifier — INFO — Total test accuracy: 0.7992
fashion-mnist-classifier_1  | 2023-09-30 18:26:17,982 — classifier — INFO — Total test F1_macro score: 0.7955895683253991
fashion-mnist-classifier_1  | 2023-09-30 18:26:17,983 — classifier — INFO — Confusion matrix:
fashion-mnist-classifier_1  | 2023-09-30 18:26:17,983 — classifier — INFO — [[806   2   9  82   7   9  66   0  19   0]
fashion-mnist-classifier_1  |  [  7 919  15  48   7   0   2   0   2   0]
fashion-mnist-classifier_1  |  [ 23   0 651  11 171   2 126   0  16   0]
fashion-mnist-classifier_1  |  [ 31   9   2 856  26   2  69   0   5   0]
fashion-mnist-classifier_1  |  [  1   1 122  51 707   1 109   0   8   0]
fashion-mnist-classifier_1  |  [  0   0   0   2   0 924   0  53   2  19]
fashion-mnist-classifier_1  |  [232   1 124  56 153   3 398   0  33   0]
fashion-mnist-classifier_1  |  [  0   0   0   0   0  67   0 863   1  69]
fashion-mnist-classifier_1  |  [  4   1   9   9   1   9  13   7 946   1]
fashion-mnist-classifier_1  |  [  0   0   0   2   0  26   0  49   1 922]]
fashion-mnist-classifier_1  | .
fashion-mnist-classifier_1  | ----------------------------------------------------------------------
fashion-mnist-classifier_1  | Ran 1 test in 3.306s
fashion-mnist-classifier_1  | 
fashion-mnist-classifier_1  | OK
fashion-mnist-classifier_1  | 2023-09-30 18:26:17,986 — __main__ — INFO — Tests passed!
fashion-mnist-classifier_1  | Name                                      Stmts   Miss  Cover   Missing
fashion-mnist-classifier_1  | -----------------------------------------------------------------------
fashion-mnist-classifier_1  | src/ansible_credential_utils.py               8      2    75%   11-12
fashion-mnist-classifier_1  | src/classifier.py                           105     40    62%   36, 56-58, 77, 80-126
fashion-mnist-classifier_1  | src/dataset_utils.py                         15      0   100%
fashion-mnist-classifier_1  | src/db_utils.py                              38      2    95%   34, 49
fashion-mnist-classifier_1  | src/fashion_mnist_classifier.py              19      0   100%
fashion-mnist-classifier_1  | src/logger.py                                26      0   100%
fashion-mnist-classifier_1  | src/model.py                                 10      0   100%
fashion-mnist-classifier_1  | src/unit_tests/test_training_results.py      42      1    98%   58
fashion-mnist-classifier_1  | -----------------------------------------------------------------------
fashion-mnist-classifier_1  | TOTAL                                       263     45    83%
big-data-hw_fashion-mnist-classifier_1 exited with code 0
Stopping database                               ... 
Stopping database                               ... done
Aborting on container exit...
```