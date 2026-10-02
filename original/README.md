# Original code (2023)

`original_code.py` is the script exactly as it appears in the appendix of my 3rd-year report, written as one long file.

It won't run as-is: it `chdir`s into output folders that aren't in this repo, and it expects the Amazon CSVs in the working directory. The code in `src/` is the cleaned-up version of the same thing, split into one script per experiment with a few small bug fixes. It gives the same results.
