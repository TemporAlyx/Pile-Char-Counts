# Pile-Char-Counts
counts unique unicode chars within the pile dataset (deduplicated)

I couldn't find anywhere that had these stats, so I wrote my own version. 
Turns out the difference between counting chars in a 10k char dataset is very different from trying to do that with a 800 billion dataset. Who knew?

This repo includes both the code to download/load the dataset via huggingface, as well as map a function on it in batches that counts up the unique unicode chars.
However, if you don't want to spend an afternoon downloading and processing anything, I also include the final aggregated counts as an npy file, which is loaded by default by the notebook for visualization.

Here is a simple plot output from the included notebook:
![Plot](plot.png)
