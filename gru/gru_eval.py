import os
import tensorflow as tf
import argparse
import numpy as np
import time
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Embedding,
    LSTM,
    SpatialDropout1D,
    GRU,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from nltk.corpus import stopwords
import re
import warnings
import string, time
import contractions

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

DATASET = "../IMDB Dataset.csv"
TRAINED_MODEL = "gru_model.h5"
OUTPUT_FILE = "gru_time.txt"

chat_words = {
    "AFAIK": "As Far As I Know",
    "AFK": "Away From Keyboard",
    "ASAP": "As Soon As Possible",
    "ATK": "At The Keyboard",
    "ATM": "At The Moment",
    "A3": "Anytime, Anywhere, Anyplace",
    "BAK": "Back At Keyboard",
    "BBL": "Be Back Later",
    "BBS": "Be Back Soon",
    "BFN": "Bye For Now",
    "B4N": "Bye For Now",
    "BRB": "Be Right Back",
    "BRT": "Be Right There",
    "BTW": "By The Way",
    "B4": "Before",
    "CU": "See You",
    "CUL8R": "See You Later",
    "CYA": "See You",
    "FAQ": "Frequently Asked Questions",
    "FC": "Fingers Crossed",
    "FWIW": "For What It's Worth",
    "FYI": "For Your Information",
    "GAL": "Get A Life",
    "GG": "Good Game",
    "GN": "Good Night",
    "GMTA": "Great Minds Think Alike",
    "GR8": "Great!",
    "G9": "Genius",
    "IC": "I See",
    "ICQ": "I Seek you (also a chat program)",
    "ILU": "I Love You",
    "IMHO": "In My Honest/Humble Opinion",
    "IMO": "In My Opinion",
    "IOW": "In Other Words",
    "IRL": "In Real Life",
    "KISS": "Keep It Simple, Stupid",
    "LDR": "Long Distance Relationship",
    "LMAO": "Laugh My A.. Off",
    "LOL": "Laughing Out Loud",
    "LTNS": "Long Time No See",
    "L8R": "Later",
    "MTE": "My Thoughts Exactly",
    "M8": "Mate",
    "NRN": "No Reply Necessary",
    "OIC": "Oh I See",
    "PITA": "Pain In The A..",
    "PRT": "Party",
    "PRW": "Parents Are Watching",
    "QPSA": "Que Pasa?",
    "ROFL": "Rolling On The Floor Laughing",
    "ROFLOL": "Rolling On The Floor Laughing Out Loud",
    "ROTFLMAO": "Rolling On The Floor Laughing My A.. Off",
    "SK8": "Skate",
    "STATS": "Your sex and age",
    "ASL": "Age, Sex, Location",
    "THX": "Thank You",
    "TTFN": "Ta-Ta For Now!",
    "TTYL": "Talk To You Later",
    "U": "You",
    "U2": "You Too",
    "U4E": "Yours For Ever",
    "WB": "Welcome Back",
    "WTF": "What The F...",
    "WTG": "Way To Go!",
    "WUF": "Where Are You From?",
    "W8": "Wait...",
    "7K": "Sick:-D Laughter",
    "TFW": "That feeling when",
    "MFW": "My face when",
    "MRW": "My reaction when",
    "IFYP": "I feel your pain",
    "LOL": "Laughing out loud",
    "TNTL": "Trying not to laugh",
    "JK": "Just kidding",
    "IDC": "I don’t care",
    "ILY": "I love you",
    "IMU": "I miss you",
    "ADIH": "Another day in hell",
    "IDC": "I don’t care",
    "ZZZ": "Sleeping, bored, tired",
    "WYWH": "Wish you were here",
    "TIME": "Tears in my eyes",
    "BAE": "Before anyone else",
    "FIMH": "Forever in my heart",
    "BSAAW": "Big smile and a wink",
    "BWL": "Bursting with laughter",
    "LMAO": "Laughing my a** off",
    "BFF": "Best friends forever",
    "CSL": "Can’t stop laughing",
}


#####################
# Argument Parsing
#####################
def parse_args():
    parser = argparse.ArgumentParser(
        description="LSTM inference with GPU memory limitation and CPU core limitation."
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for inference. (Default = False)",
    )
    parser.add_argument(
        "--gpu_mem_limit",
        type=int,
        default=0,
        help="Limit GPU memory usage in MB. Set to 0 for no limit. (Default = 512)",
    )
    parser.add_argument(
        "--memory_growth",
        action="store_true",
        help="Enable GPU memory growth. (Default = False)",
    )
    parser.add_argument(
        "--cpu_cores",
        type=int,
        default=None,
        help="Limit the number of CPU cores for TensorFlow. Set to None for no limit.",
    )
    return parser.parse_args()


args = parse_args()

# Configurations
USE_GPU = args.gpu
GPU_MEM_LIMIT = args.gpu_mem_limit
MEMORY_GROWTH = args.memory_growth
CPU_CORES = args.cpu_cores


#####################
# GPU Setup
#####################
def setup_gpu(use_gpu=True, memory_growth=False, gpu_mem_limit=0):
    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print("Number of GPUs Available: {}".format(len(gpus)))

        # Set memory growth and memory limit
        for gpu in gpus:
            if memory_growth:
                tf.config.experimental.set_memory_growth(gpu, True)
            if gpu_mem_limit > 0:
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=gpu_mem_limit)],
                )


#####################
# CPU Setup (Limit CPU cores)
#####################
def setup_cpu(cpu_cores=None):
    if cpu_cores:
        # Limit the number of CPU cores TensorFlow uses
        tf.config.threading.set_intra_op_parallelism_threads(cpu_cores)
        tf.config.threading.set_inter_op_parallelism_threads(cpu_cores)
        print("Limiting TensorFlow to {} CPU cores.".format(cpu_cores))
    else:
        print("Using all available CPU cores.")


setup_gpu(USE_GPU, MEMORY_GROWTH, GPU_MEM_LIMIT)
setup_cpu(CPU_CORES)

#####################
# Data Preparation
#####################
# Load the dataset
# Please replace the path with the actual path to the IMDB dataset
data = pd.read_csv(DATASET)

# Text preprocessing
STOPWORDS = set(stopwords.words("english"))


def clean_text(text):
    text = text.lower()  # lowercase text
    text = " ".join(
        word for word in text.split() if word not in STOPWORDS
    )  # remove stopwords from text
    text = re.sub(r"\W", " ", text)  # Remove all the special characters
    text = re.sub(r"\s+[a-zA-Z]\s+", " ", text)  # remove all single characters
    text = re.sub(
        r"\^[a-zA-Z]\s+", " ", text
    )  # Remove single characters from the start
    text = re.sub(
        r"\s+", " ", text, flags=re.I
    )  # Substituting multiple spaces with single space
    return text


def remove_url(text):
    pattern = re.compile(r"https?://\S+|www\.\S+")
    return pattern.sub(r"", text)


def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))


def chat_conversion(text):
    new_text = []
    for w in text.split():
        if w.upper() in chat_words:
            new_text.append(chat_words[w.upper()])
        else:
            new_text.append(w)
    return " ".join(new_text)


def remove_emoji(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", text)


def expand_contractions(text):
    expanded_text = contractions.fix(text)
    return expanded_text


data["review"] = data["review"].str.lower()
data["review"] = data["review"].apply(remove_url)
data["review"] = data["review"].apply(remove_punctuation)
data["review"] = data["review"].apply(chat_conversion)
data["review"] = data["review"].apply(clean_text)
data["review"] = data["review"].apply(remove_emoji)
data["review"] = data["review"].apply(expand_contractions)

#####################
# Tokenization and Seqeunce Padding
#####################
# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 250
# This is fixed.
EMBEDDING_DIM = 100

tokenizer = Tokenizer(
    num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True
)
tokenizer.fit_on_texts(data["review"].values)
word_index = tokenizer.word_index
print("Found %s unique tokens." % len(word_index))

X = tokenizer.texts_to_sequences(data["review"].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
# print("Shape of data tensor:", X.shape)


#####################
# Prepare the labels
#####################
# Converting categorical labels to numbers.
Y = pd.get_dummies(data["sentiment"]).values
# print("Shape of label tensor:", Y.shape)

# Train test split
split_idx = int(0.8 * len(X))  # 80% training, 20% testing
X_train, X_test = X[:split_idx], X[split_idx:]
Y_train, Y_test = Y[:split_idx], Y[split_idx:]

# print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

#####################
# LSTM Model Creation for Inference
#####################
loaded_model = load_model(TRAINED_MODEL)

#####################
# Inference Timing
#####################
start_time = time.time()
accr = loaded_model.evaluate(X_test, Y_test)
end_time = time.time()

the_time = end_time - start_time

print("Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}".format(accr[0], accr[1]))
print("Time taken for inference: {:.2f} seconds".format(the_time))


with open("gru_time.txt", "a") as file:
    file.write(
        "\nEvaluate with options. GPU({}) / GMEM({}) / MEMG({}) / CPU({})".format(
            USE_GPU, GPU_MEM_LIMIT, MEMORY_GROWTH, CPU_CORES
        )
    )
    file.write("\n{}".format(the_time))
    file.write("\n")
