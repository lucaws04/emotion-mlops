CLI Module Documentation
========================

This module provides a command-line interface (CLI) to perform emotion prediction
using a fine-tuned BERT-based model.

Run the CLI with the ``predict`` command and either `--text` or `--file`.

Functions
---------
.. autofunction:: src.cli.cli.main

.. autofunction:: src.cli.cli.predict

How to Use
----------

.. code-block:: bash

    python cli.py predict --text "I am so happy today!"
    python cli.py predict --file path/to/file.txt
    python cli.py predict --file input.txt --verbose

CLI Options
-----------

.. option:: --text, -t TEXT

    A single input sentence for emotion prediction.

.. option:: --file, -f FILE

    A path to a `.txt` file containing input sentences (one per line).

.. option:: --verbose, -v

    If set, enables debug logging and prints additional output.



Model Components
----------------

These components are imported from ``src.inference.cli_inference``.

``load_emotion_model``
----------------------

.. autofunction:: src.inference.cli_inference.load_emotion_model


``predict_emotion_single``
--------------------------

.. autofunction:: src.inference.cli_inference.predict_emotion_single

``predict_from_input``
----------------------

.. autofunction:: src.inference.cli_inference.predict_from_input
