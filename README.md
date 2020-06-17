# Demonstrate issue with tensorflow_text and TFX

To run the example, copy the file to ~/ (home directory)
`python pipeline.py`

The pipeline is ingesting a dummy csv dataset.
The TFX transformer uses `preprocess_fn` in `utils.py`
Within `preprocess_fn`, it's simply trying to tokenize a `tf.constant` string.

To demonstrate the issue, set `direct_num_workers` to 0 which will tell the runner to run the task on as many workers as necessary.
It works fine with `direct_num_workers = 1` which leads to the belief that the error is a result of inproper import/compile for multiple workers.
