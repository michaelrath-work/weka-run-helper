# Weka execution helper

Weka has a highly sophisticated command line usage. This wrapper simplifies to run weka from command line

## Basic usage

* Run from command line, use `--help` to get all options

    ```bash
    python run_weka.py --train-file train.arff --test-file test.arff --model-file model.pmml
    ```

* **IMPORTANT** per default, `weka` is not called, unless explicitly specified with `--run` option

    ```bash
    python run_weka.py <options> --run

    python run_weka.py --settings-input-file pig.json --run
    ```

## A worked example

1. Install [weka](https://www.cs.waikato.ac.nz/~ml/weka/) and locate the `weka.jar` file. This path needs be specified with argument `--weka-jar`

2. *Train* classifier `zeroR` with attribute configuration `auto` on training set `my_train.arff` (also used for testing) and save trained model in `saved_model.pmml`

    ```bash
    python run_weka.py -e train -c zeroR --attribute-config=auto --train-file my_train.arff --test-file my_train.arff --model-file saved_model.pmml --run
    ```

3. *Predict* the samples in `my_predict.arff` using the previously saved model `saved_model.pmml` and store the prediction output in `prediction.csv`

    ```bash
    python run_weka.py -e predict --test-file my_predict.arff --model-file saved_model.pmml --prediction-output-file prediction.csv --run
    ```

## Settings handling

* Settings can be saved with

    ```bash
    python run_weka.py --settings-output-file settings.json
    ```

* Settings can be loaded with

    ```bash
    python run_weka.py --settings-input-file settings.json
    ```

* It is also possible, to *overwrite* settings from file on command line, e.g. change `sub-sample-seed` to 10, independent from value specified in file

    1. Defaults defined in [`DEFAULT_SETTINGS`](run_weka.py) in source code (**lowest** precedence)
    2. Settings loaded from settings file
    3. Settings specified on command line (**highest** precedence)

    ```bash
    python run_weka.py --settings-input-file settings.json --sub-sample-seed 10
    ```

* Settings file rewriting

    ```bash
    python run_weka.py --settings-input-file in.json --settings-output-file out.json --classifier j48
    ```
