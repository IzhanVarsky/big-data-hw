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
database                    | syncing data to disk ... initdb: warning: enabling "trust" authentication for local connections
database                    | initdb: hint: You can change this by editing pg_hba.conf or using the option -A, or --auth-local and --auth-host, the next time you run initdb.
database                    | ok
database                    | 
database                    | 
database                    | Success. You can now start the database server using:
database                    | 
database                    |     pg_ctl -D /var/lib/***ql/data -l logfile start
database                    | 
database                    | waiting for server to start....2023-09-30 16:22:03.665 UTC [48] LOG:  starting PostgreSQL 16.0 (Debian 16.0-1.pgdg120+1) on x86_64-pc-linux-gnu, compiled by gcc (Debian 12.2.0-14) 12.2.0, 64-bit
database                    | 2023-09-30 16:22:03.666 UTC [48] LOG:  listening on Unix socket "/var/run/***ql/.s.PGSQL.5432"
database                    | 2023-09-30 16:22:03.669 UTC [51] LOG:  database system was shut down at 2023-09-30 16:22:03 UTC
database                    | 2023-09-30 16:22:03.673 UTC [48] LOG:  database system is ready to accept connections
database                    |  done
database                    | server started
database                    | 
database                    | /usr/local/bin/docker-entrypoint.sh: running /docker-entrypoint-initdb.d/init.sql
database                    | CREATE TABLE
database                    | INSERT 0 1
database                    | CREATE TABLE
database                    | 
database                    | 
database                    | 2023-09-30 16:22:03.828 UTC [48] LOG:  received fast shutdown request
database                    | waiting for server to shut down....2023-09-30 16:22:03.829 UTC [48] LOG:  aborting any active transactions
database                    | 2023-09-30 16:22:03.831 UTC [48] LOG:  background worker "logical replication launcher" (PID 54) exited with exit code 1
database                    | 2023-09-30 16:22:03.832 UTC [49] LOG:  shutting down
database                    | 2023-09-30 16:22:03.833 UTC [49] LOG:  checkpoint starting: shutdown immediate
database                    | 2023-09-30 16:22:03.838 UTC [49] LOG:  checkpoint complete: wrote 65 buffers (0.4%); 0 WAL file(s) added, 0 removed, 0 recycled; write=0.002 s, sync=0.002 s, total=0.006 s; sync files=48, longest=0.002 s, average=0.001 s; distance=207 kB, estimate=207 kB; lsn=0/151E718, redo lsn=0/151E718
database                    | 2023-09-30 16:22:03.841 UTC [48] LOG:  database system is shut down
database                    |  done
database                    | server stopped
database                    | 
database                    | PostgreSQL init process complete; ready for start up.
database                    | 
database                    | 2023-09-30 16:22:03.955 UTC [1] LOG:  starting PostgreSQL 16.0 (Debian 16.0-1.pgdg120+1) on x86_64-pc-linux-gnu, compiled by gcc (Debian 12.2.0-14) 12.2.0, 64-bit
database                    | 2023-09-30 16:22:03.955 UTC [1] LOG:  listening on IPv4 address "0.0.0.0", port 5432
database                    | 2023-09-30 16:22:03.955 UTC [1] LOG:  listening on IPv6 address "::", port 5432
database                    | 2023-09-30 16:22:03.956 UTC [1] LOG:  listening on Unix socket "/var/run/***ql/.s.PGSQL.5432"
database                    | 2023-09-30 16:22:03.959 UTC [64] LOG:  database system was shut down at 2023-09-30 16:22:03 UTC
database                    | 2023-09-30 16:22:03.964 UTC [1] LOG:  database system is ready to accept connections
Creating big-data-hw_fashion-mnist-classifier_1 ... done
fashion-mnist-classifier_1  | 2023-09-30 16:22:10,269 — __main__ — INFO — Testing datasets
fashion-mnist-classifier_1  | 2023-09-30 16:22:10,316 — __main__ — INFO — Datasets collected
fashion-mnist-classifier_1  | 2023-09-30 16:22:10,316 — __main__ — INFO — Testing datasets len...
fashion-mnist-classifier_1  | 2023-09-30 16:22:10,316 — __main__ — INFO — Testing datasets len passed!
fashion-mnist-classifier_1  | .2023-09-30 16:22:10,317 — __main__ — INFO — Testing datasets
fashion-mnist-classifier_1  | 2023-09-30 16:22:10,361 — __main__ — INFO — Datasets collected
fashion-mnist-classifier_1  | 2023-09-30 16:22:10,361 — __main__ — INFO — Testing datasets types...
fashion-mnist-classifier_1  | 2023-09-30 16:22:10,361 — __main__ — INFO — Testing datasets types passed!
fashion-mnist-classifier_1  | .
fashion-mnist-classifier_1  | ----------------------------------------------------------------------
fashion-mnist-classifier_1  | Ran 2 tests in 0.093s
fashion-mnist-classifier_1  | 
fashion-mnist-classifier_1  | OK
fashion-mnist-classifier_1  | 2023-09-30 16:22:13,375 — __main__ — INFO — Results table:
fashion-mnist-classifier_1  | 2023-09-30 16:22:13,376 — __main__ — INFO — --------------------------------------------------------------
fashion-mnist-classifier_1  |  id | model_path                                              
fashion-mnist-classifier_1  | ----+---------------------------------------------------------
fashion-mnist-classifier_1  |   1 | checkpoints/CNNModel_FashionMNIST_best_metric_model.pth 
fashion-mnist-classifier_1  | --------------------------------------------------------------
fashion-mnist-classifier_1  | (1 row)
fashion-mnist-classifier_1  | 
fashion-mnist-classifier_1  | 2023-09-30 16:22:13,380 — __main__ — INFO — Collecting dataloaders
fashion-mnist-classifier_1  | 2023-09-30 16:22:13,423 — __main__ — INFO — Loading classifier
fashion-mnist-classifier_1  | 2023-09-30 16:22:13,435 — __main__ — INFO — Checkpoint loaded
fashion-mnist-classifier_1  | 2023-09-30 16:22:13,435 — __main__ — INFO — Testing classifier metrics...
fashion-mnist-classifier_1  | 2023-09-30 16:22:13,436 — classifier — INFO — *************************
fashion-mnist-classifier_1  | 2023-09-30 16:22:13,436 — classifier — INFO — >> Testing CNNModel network
fashion-mnist-classifier_1  | 
  0%|          | 0/40 [00:00<?, ?it/s]
  5%|▌         | 2/40 [00:00<00:01, 19.41it/s]
 12%|█▎        | 5/40 [00:00<00:02, 15.97it/s]
 18%|█▊        | 7/40 [00:00<00:01, 16.91it/s]
 25%|██▌       | 10/40 [00:00<00:01, 18.88it/s]
 32%|███▎      | 13/40 [00:00<00:01, 20.38it/s]
 40%|████      | 16/40 [00:00<00:01, 21.42it/s]
 48%|████▊     | 19/40 [00:00<00:00, 22.08it/s]
 55%|█████▌    | 22/40 [00:01<00:00, 22.48it/s]
 62%|██████▎   | 25/40 [00:01<00:00, 22.78it/s]
 70%|███████   | 28/40 [00:01<00:00, 23.01it/s]
 78%|███████▊  | 31/40 [00:01<00:00, 22.95it/s]
 85%|████████▌ | 34/40 [00:01<00:00, 23.13it/s]
 92%|█████████▎| 37/40 [00:01<00:00, 22.68it/s]
100%|██████████| 40/40 [00:01<00:00, 22.14it/s]
fashion-mnist-classifier_1  | 2023-09-30 16:22:16,218 — classifier — INFO — Total test loss: 0.5603217876434327
fashion-mnist-classifier_1  | 2023-09-30 16:22:16,218 — classifier — INFO — Total test accuracy: 0.7992
fashion-mnist-classifier_1  | 2023-09-30 16:22:16,218 — classifier — INFO — Total test F1_macro score: 0.7955895683253991
fashion-mnist-classifier_1  | 2023-09-30 16:22:16,218 — classifier — INFO — Confusion matrix:
fashion-mnist-classifier_1  | 2023-09-30 16:22:16,218 — classifier — INFO — [[806   2   9  82   7   9  66   0  19   0]
fashion-mnist-classifier_1  |  [  7 919  15  48   7   0   2   0   2   0]
fashion-mnist-classifier_1  |  [ 23   0 651  11 171   2 126   0  16   0]
fashion-mnist-classifier_1  |  [ 31   9   2 856  26   2  69   0   5   0]
fashion-mnist-classifier_1  |  [  1   1 122  51 707   1 109   0   8   0]
fashion-mnist-classifier_1  |  [  0   0   0   2   0 924   0  53   2  19]
fashion-mnist-classifier_1  |  [232   1 124  56 153   3 398   0  33   0]
fashion-mnist-classifier_1  |  [  0   0   0   0   0  67   0 863   1  69]
fashion-mnist-classifier_1  |  [  4   1   9   9   1   9  13   7 946   1]
fashion-mnist-classifier_1  |  [  0   0   0   2   0  26   0  49   1 922]]
fashion-mnist-classifier_1  | 2023-09-30 16:22:16,222 — __main__ — INFO — Tests passed!
fashion-mnist-classifier_1  | .
fashion-mnist-classifier_1  | ----------------------------------------------------------------------
fashion-mnist-classifier_1  | Ran 1 test in 2.842s
fashion-mnist-classifier_1  | 
fashion-mnist-classifier_1  | OK
fashion-mnist-classifier_1  | Name                                      Stmts   Miss  Cover   Missing
fashion-mnist-classifier_1  | -----------------------------------------------------------------------
fashion-mnist-classifier_1  | src/classifier.py                           105     40    62%   36, 56-58, 77, 80-126
fashion-mnist-classifier_1  | src/dataset_utils.py                         15      0   100%
fashion-mnist-classifier_1  | src/db_utils.py                              24      0   100%
fashion-mnist-classifier_1  | src/fashion_mnist_classifier.py              19      0   100%
fashion-mnist-classifier_1  | src/logger.py                                26      0   100%
fashion-mnist-classifier_1  | src/model.py                                 10      0   100%
fashion-mnist-classifier_1  | src/unit_tests/test_training_results.py      39      0   100%
fashion-mnist-classifier_1  | -----------------------------------------------------------------------
fashion-mnist-classifier_1  | TOTAL                                       238     40    83%
big-data-hw_fashion-mnist-classifier_1 exited with code 0
Stopping database                               ... 
Stopping database                               ... done
Aborting on container exit...
```