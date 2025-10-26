Consumes a CSV with Puerto Rico business metadata and returns an output CSV and decides which ones have sufficient evidence of existence.
The results are encoded in an output CSV with one row called `Keep row`. `True` entries indicate a business should be considered active.

Have the CSV of Puerto Rico business entries in the root folder named `businesses.csv`

Run the code in a virtual environment with `python3 main.py`

`src/config.py` contains a number of parameters that the user can tune.

The output CSV will be saved in the root directory under `businesses_to_keep.csv` (unless name is modified in `src/config.py`)