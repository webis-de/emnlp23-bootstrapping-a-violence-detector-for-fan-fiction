# Trigger Warning Classification

## General

All single scripts are cli programs, i.e. you can call them with "-h" to print a help. 

Example:
```
python -m trigger_warning_classification.preprocess -h
```

Prints:
```bash
$ python -m trigger_warning_classification.preprocess -h
usage: preprocess.py [-h] input_file output_train_file [output_test_file]

positional arguments:
  input_file         path to the input data (jsonlines)
  output_train_file  path to the output train file (tsv)
  output_test_file   [optional] path to the output test file (tsv)

optional arguments:
  -h, --help         show this help message and exit
```

Download the data at [zenodo](https://zenodo.org/records/10036479)

## Classification Scripts

This projects contains the following scripts:

1. preprocess.py: Converts the input jsonlines file, applies some preprocessing, and creates train and test splits.
   *The last argument is optional: If you omit `output_test_file` no splits are created and 
   you just convert the input file into a preprocessed tsv.*
   ```
   python -m trigger_warning_classification.preprocess /mnt/ceph/storage/data-in-progress/data-research/computational-ethics/trigger-warnings/best_works.jsonl works-preprocessed-train.tsv works-preprocessed-test.tsv
   ```

2. hyperparameters_svm.py: Explores several hyperparameters for the SVM (only using the train set).
   ```
   python -m trigger_warning_classification.hyperparameters_svm works-preprocessed-train.tsv hyperparameters-svm-results.tsv
   ```

3. train_svm.py: Trains a model on the `works-preprocessed-train.tsv` and evaluates it on `works-preprocessed-test.tsv` (i.e. on the outputs of preprocess.py).
   The resulting model is saved as `model.pkl`.
   ```
   python -m trigger_warning_classification.train_svm works-preprocessed-train.tsv works-preprocessed-test.tsv model.pkl
   ```
  
4. predict.py: With the trained model you can obtain predictions on new data. 
   The expected tsv format is the same as in the training step.
   ```
   python -m trigger_warning_classification.predict works-preprocessed-train.tsv results.tsv model.pkl
   ```
   The results file contains the documents in the same order with predictions and confidence estimates 
   but without content and title fields.

### Predicting new samples

If you want use the model to predict on new unseen data:

1. Call preprocess (without the `output_test_file` parameter).
2. Call classify with the tsv file from preprocessing and the model from the previous training.
