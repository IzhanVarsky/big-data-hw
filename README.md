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

#### Дополнительная информация

* Пароли для авторизации на DockerHub и на удаленном сервере были сохранены с использованием GitHub Secrets
* Ссылка на загруженный Docker
  образ: https://hub.docker.com/repository/docker/izhanvarsky/bigdata-hw1-fashion-mnist-classifier
* Результаты тестирования можно найти
  здесь: https://github.com/IzhanVarsky/big-data-hw1/actions/workflows/docker_run_tests.yaml
* Пример результатов тестирования:

```
fashion-mnist-classifier_1  | ----------------------------------------------------------------------
fashion-mnist-classifier_1  | Ran 2 tests in 0.141s
fashion-mnist-classifier_1  | 
fashion-mnist-classifier_1  | OK
fashion-mnist-classifier_1  | *************************
fashion-mnist-classifier_1  | >> Testing CNNModel network
fashion-mnist-classifier_1  | 
  0%|          | 0/40 [00:00<?, ?it/s]
  5%|▌         | 2/40 [00:00<00:02, 18.33it/s]
 10%|█         | 4/40 [00:00<00:01, 18.79it/s]
 18%|█▊        | 7/40 [00:00<00:01, 19.47it/s]
 25%|██▌       | 10/40 [00:00<00:01, 20.13it/s]
 32%|███▎      | 13/40 [00:00<00:01, 20.42it/s]
 40%|████      | 16/40 [00:00<00:01, 20.61it/s]
 48%|████▊     | 19/40 [00:00<00:01, 20.79it/s]
 55%|█████▌    | 22/40 [00:01<00:00, 20.84it/s]
 62%|██████▎   | 25/40 [00:01<00:00, 21.04it/s]
 70%|███████   | 28/40 [00:01<00:00, 17.85it/s]
 78%|███████▊  | 31/40 [00:01<00:00, 18.63it/s]
 85%|████████▌ | 34/40 [00:01<00:00, 19.34it/s]
 92%|█████████▎| 37/40 [00:01<00:00, 20.11it/s]
100%|██████████| 40/40 [00:01<00:00, 22.17it/s]
100%|██████████| 40/40 [00:01<00:00, 20.32it/s]
fashion-mnist-classifier_1  | Total test loss: 0.5603217876434327
fashion-mnist-classifier_1  | Total test accuracy: 0.7992
fashion-mnist-classifier_1  | Total test F1_macro score: 0.7955895683253991
fashion-mnist-classifier_1  | Confusion matrix:
fashion-mnist-classifier_1  | [[806   2   9  82   7   9  66   0  19   0]
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
fashion-mnist-classifier_1  | Ran 1 test in 2.956s
fashion-mnist-classifier_1  | 
fashion-mnist-classifier_1  | OK
fashion-mnist-classifier_1  | Name                                      Stmts   Miss  Cover   Missing
fashion-mnist-classifier_1  | -----------------------------------------------------------------------
fashion-mnist-classifier_1  | src/classifier.py                           103     40    61%   32, 52-54, 73, 76-122
fashion-mnist-classifier_1  | src/dataset_utils.py                         15      0   100%
fashion-mnist-classifier_1  | src/fashion_mnist_classifier.py              19      0   100%
fashion-mnist-classifier_1  | src/model.py                                 10      0   100%
fashion-mnist-classifier_1  | src/unit_tests/test_training_results.py      22      0   100%
fashion-mnist-classifier_1  | -----------------------------------------------------------------------
fashion-mnist-classifier_1  | TOTAL                                       169     40    76%
big-data-hw1_fashion-mnist-classifier_1 exited with code 0
Aborting on container exit...
```