import numpy as np
import gc, os

from datasets import load_dataset
from datasets import VerificationMode

the_pile = None
cache_dir = "S:/Datasets/" # change to your cache dir or None if you want to use the default
batch_size = 10000 # too much lower leads to more storage space for the aggregated counts
perc_at_once = 5 # how many percent of the pile to process at once, running it all at once leads to memory issues (not sure why as its supposed to be dynamically allocated)
n_cpus = 20 # number of cpus to use for processing
# dropping either n_cpus or perc_at_once could reduce memory usage, I do not reccomend lowering batch size though, as the total filecount was ~112GB

# unicode_values = np.zeros((1114112,), dtype=np.int64)

def ext_count_func_batched(texts):
    processed_text = np.array([ord(x) for x in list("".join(texts['text']))])
    loc_unicode_values = np.bincount(processed_text, minlength=1114112)
    random_name = os.getcwd()+"/charcounts/unicode_counts_"+str(np.random.randint(0, 1e18, dtype=np.int64))+".npy"
    # check if file exists
    while os.path.exists(random_name): # note: should rewrite paths to be os independent
        random_name = os.getcwd()+"/charcounts/unicode_counts_"+str(np.random.randint(0, 1e18, dtype=np.int64))+".npy"
    # save
    np.save(random_name, loc_unicode_values)
    return None

def load_perc_pile(start, end):
    global the_pile
    the_pile = load_dataset("EleutherAI/raw_deduplicated_pile", split="train["+str(start)+"%:"+str(end)+"%]", 
                            cache_dir=cache_dir, verification_mode=VerificationMode.NO_CHECKS,)

if __name__ == "__main__":

    for i in range(0, 100, 5):
        print("Loading pile from "+str(i)+" to "+str(i+5))
        load_perc_pile(i, i+5)

        # process
        the_pile.map(ext_count_func_batched, batched=True, batch_size=batch_size, num_proc=20)

        # clear memory
        the_pile = None
        gc.collect()