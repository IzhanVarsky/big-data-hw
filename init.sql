CREATE TABLE model_weights
(
    id         SERIAL NOT NULL PRIMARY KEY,
    model_path TEXT   NOT NULL
);

INSERT INTO model_weights (model_path)
VALUES ('checkpoints/CNNModel_FashionMNIST_best_metric_model.pth');

CREATE TABLE model_results
(
    id         SERIAL NOT NULL PRIMARY KEY,
    epoch_loss TEXT   NOT NULL,
    epoch_acc  TEXT   NOT NULL,
    f1_macro   TEXT   NOT NULL
);
