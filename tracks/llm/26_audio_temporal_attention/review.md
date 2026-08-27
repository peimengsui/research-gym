# Review

After the tests pass, explain:

- how original sample lengths become valid STFT frame counts
- why a partially valid temporal patch is treated as invalid
- why invalid query rows can create NaNs
- why masking only attention keys is insufficient for padded outputs
- why valid outputs should not change when padded samples change

Compare your reshape and mask logic with `solution.locked.py`.
