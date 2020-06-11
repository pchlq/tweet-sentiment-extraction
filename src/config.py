import tokenizers
import os
import sentencepiece as spm
import sentencepiece_pb2
import transformers

# from transformers import AlbertConfig


class SentencePieceTokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(os.path.join(model_path, "spiece.model"))

    def encode(self, sentence):
        spt = sentencepiece_pb2.SentencePieceText()
        spt.ParseFromString(self.sp.encode_as_serialized_proto(sentence))
        offsets = []
        tokens = []
        for piece in spt.pieces:
            tokens.append(piece.id)
            offsets.append((piece.begin, piece.end))
        return tokens, offsets


model_type = "roberta_base"
# model_type='albert_large_v2'
# model_type='gpt2_medium_xlm'
# model_type='albert_base_v2'  #albert-xxlarge-v1 albert-large-v1

DROPOUT = 0.15
LEARNING_RATE = 3e-5
MAX_LEN = 160
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 4
EPOCHS = 5
MODELS_PATH = "/home/pchlq/workspace/models_nlp"
MODEL_DIR = f"{MODELS_PATH}/{model_type}/"
TRAINING_FILE = "../input/train_folds.csv"
TOKENIZER = SentencePieceTokenizer(MODEL_DIR)

# TOKENIZER = tokenizers.ByteLevelBPETokenizer(
#     vocab_file=f"{MODEL_DIR}/vocab.json",
#     merges_file=f"{MODEL_DIR}/merges.txt",
#     lowercase=True,
#     add_prefix_space=True
# )

# albert_base_configuration = transformers.AlbertConfig(
#     hidden_size=768,
#     num_attention_heads=12,
#     intermediate_size=3072,
# )
