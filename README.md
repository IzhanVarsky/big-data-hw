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
Creating network "big-data-hw1_default" with the default driver
Creating big-data-hw1_fashion-mnist-classifier_1 ... 
Creating big-data-hw1_fashion-mnist-classifier_1 ... done
Attaching to big-data-hw1_fashion-mnist-classifier_1
fashion-mnist-classifier_1  | 2023-09-12 00:39:30,474:root:INFO:`>> Testing datasets`
fashion-mnist-classifier_1  | 2023-09-12 00:39:30,568:root:INFO:`>> Datasets collected`
fashion-mnist-classifier_1  | 2023-09-12 00:39:30,568:root:INFO:`>> Testing datasets len...`
fashion-mnist-classifier_1  | 2023-09-12 00:39:30,569:root:INFO:`>> Testing datasets len passed!`
fashion-mnist-classifier_1  | .2023-09-12 00:39:30,569:root:INFO:`>> Testing datasets`
fashion-mnist-classifier_1  | 2023-09-12 00:39:30,618:root:INFO:`>> Datasets collected`
fashion-mnist-classifier_1  | 2023-09-12 00:39:30,619:root:INFO:`>> Testing datasets types...`
fashion-mnist-classifier_1  | 2023-09-12 00:39:30,619:root:INFO:`>> Testing datasets types passed!`
fashion-mnist-classifier_1  | .
fashion-mnist-classifier_1  | ----------------------------------------------------------------------
fashion-mnist-classifier_1  | Ran 2 tests in 0.145s
fashion-mnist-classifier_1  | 
fashion-mnist-classifier_1  | OK
fashion-mnist-classifier_1  | 2023-09-12 00:39:33,836:root:INFO:`>> Collecting dataloaders`
fashion-mnist-classifier_1  | 2023-09-12 00:39:33,887:root:INFO:`>> Loading classifier`
fashion-mnist-classifier_1  | 2023-09-12 00:39:33,905:root:INFO:`>> Checkpoint loaded`
fashion-mnist-classifier_1  | 2023-09-12 00:39:33,905:root:INFO:`>> Testing classifier metrics...`
fashion-mnist-classifier_1  | 2023-09-12 00:39:33,905:root:INFO:`>> *************************`
fashion-mnist-classifier_1  | 2023-09-12 00:39:33,905:root:INFO:`>> >> Testing CNNModel network`
fashion-mnist-classifier_1  | 
  0%|          | 0/40 [00:00<?, ?it/s]
  5%|▌         | 2/40 [00:00<00:02, 17.98it/s]
 12%|█▎        | 5/40 [00:00<00:01, 19.19it/s]
 20%|██        | 8/40 [00:00<00:01, 19.75it/s]
 25%|██▌       | 10/40 [00:00<00:01, 19.75it/s]
 30%|███       | 12/40 [00:00<00:01, 19.72it/s]
 35%|███▌      | 14/40 [00:00<00:01, 18.61it/s]
 42%|████▎     | 17/40 [00:00<00:01, 19.25it/s]
 48%|████▊     | 19/40 [00:00<00:01, 19.37it/s]
 55%|█████▌    | 22/40 [00:01<00:00, 19.46it/s]
 62%|██████▎   | 25/40 [00:01<00:00, 20.03it/s]
 68%|██████▊   | 27/40 [00:01<00:00, 19.84it/s]
 72%|███████▎  | 29/40 [00:01<00:00, 19.70it/s]
 80%|████████  | 32/40 [00:01<00:00, 19.99it/s]
 88%|████████▊ | 35/40 [00:01<00:00, 20.36it/s]
 95%|█████████▌| 38/40 [00:01<00:00, 20.58it/s]
100%|██████████| 40/40 [00:01<00:00, 20.30it/s]
fashion-mnist-classifier_1  | 2023-09-12 00:39:36,998:root:INFO:`>> Total test loss: 0.5603217876434327`
fashion-mnist-classifier_1  | 2023-09-12 00:39:36,999:root:INFO:`>> Total test accuracy: 0.7992`
fashion-mnist-classifier_1  | 2023-09-12 00:39:36,999:root:INFO:`>> Total test F1_macro score: 0.7955895683253991`
fashion-mnist-classifier_1  | 2023-09-12 00:39:36,999:root:INFO:`>> Confusion matrix:`
fashion-mnist-classifier_1  | 2023-09-12 00:39:36,999:root:INFO:`>> [[806   2   9  82   7   9  66   0  19   0]
fashion-mnist-classifier_1  |  [  7 919  15  48   7   0   2   0   2   0]
fashion-mnist-classifier_1  |  [ 23   0 651  11 171   2 126   0  16   0]
fashion-mnist-classifier_1  |  [ 31   9   2 856  26   2  69   0   5   0]
fashion-mnist-classifier_1  |  [  1   1 122  51 707   1 109   0   8   0]
fashion-mnist-classifier_1  |  [  0   0   0   2   0 924   0  53   2  19]
fashion-mnist-classifier_1  |  [232   1 124  56 153   3 398   0  33   0]
fashion-mnist-classifier_1  |  [  0   0   0   0   0  67   0 863   1  69]
fashion-mnist-classifier_1  |  [  4   1   9   9   1   9  13   7 946   1]
fashion-mnist-classifier_1  |  [  0   0   0   2   0  26   0  49   1 922]]`
fashion-mnist-classifier_1  | 2023-09-12 00:39:37,000:root:INFO:`>> Tests passed!`
fashion-mnist-classifier_1  | .
fashion-mnist-classifier_1  | ----------------------------------------------------------------------
fashion-mnist-classifier_1  | Ran 1 test in 3.165s
fashion-mnist-classifier_1  | 
fashion-mnist-classifier_1  | OK
fashion-mnist-classifier_1  | Name                                      Stmts   Miss  Cover   Missing
fashion-mnist-classifier_1  | -----------------------------------------------------------------------
fashion-mnist-classifier_1  | src/classifier.py                           105     40    62%   40, 60-62, 81, 84-130
fashion-mnist-classifier_1  | src/dataset_utils.py                         15      0   100%
fashion-mnist-classifier_1  | src/fashion_mnist_classifier.py              19      0   100%
fashion-mnist-classifier_1  | src/model.py                                 10      0   100%
fashion-mnist-classifier_1  | src/unit_tests/test_training_results.py      29      0   100%
fashion-mnist-classifier_1  | -----------------------------------------------------------------------
fashion-mnist-classifier_1  | TOTAL                                       178     40    78%
big-data-hw1_fashion-mnist-classifier_1 exited with code 0
Aborting on container exit...
```