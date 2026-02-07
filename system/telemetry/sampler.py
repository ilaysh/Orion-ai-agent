
# sampler.py
import random
SAMPLE_RATE = 1.0  # Adjust later (1.0 = 100%)
def allow_sample():
    return random.random() <= SAMPLE_RATE
