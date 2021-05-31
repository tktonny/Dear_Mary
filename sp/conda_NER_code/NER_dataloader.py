import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import Vocab
from collections import Counter
import re
from typing import List, Tuple, Dict, Optional, Any

Sentence = List[Tuple[str, str]] 

def read_data_file(
    datapath: str,
) -> Tuple[List[Sentence], Dict[str, int], Dict[str, int]]:
    """
    Read and preprocess input data from the file `datapath`.
    Example:
    ```
        sentences, word_cnt, tag_cnt = read_data_file("eng.train")
    ```
    Return values:
        `sentences`: a list of sentences, including words and NER tags
        `word_cnt`: a Counter object, the number of occurrences of each word
        `tag_cnt`: a Counter object, the number of occurences of each NER tag
    """
    sentences: List[Sentence] = []
    word_cnt: Dict[str, int] = Counter()
    tag_cnt: Dict[str, int] = Counter()

    for sentence_txt in open(datapath).read().split("\n\n"):
        if "DOCSTART" in sentence_txt:
            # Ignore dummy sentences at the begining of each document.
            continue
        # Read a new sentence
        sentences.append([])
        for token in sentence_txt.split("\n"):
            w, _, _, t = token.split()
            # Replace all digits with "0" to reduce out-of-vocabulary words
            w = re.sub("\d", "0", w)
            word_cnt[w] += 1
            tag_cnt[t] += 1
            sentences[-1].append((w, t))

    return sentences, word_cnt, tag_cnt


class SequenceDataset(Dataset):
    """
    Each data example is a sentence, including its words and NER tags.
    """

    def __init__(
        self, datapath: str, vocab: Optional[Dict[str, Vocab]] = None
    ) -> None:
        """
        Initialize the dataset by reading from datapath.
        """
        super().__init__()
        self.sentences: List[Sentence] = []
        UNKNOWN = "<UNKNOWN>"
        PAD = "<PAD>"  # Special token used for padding 

        print("Loading data from %s" % datapath)
        self.sentences, word_cnt, tag_cnt = read_data_file(datapath)
        print("%d sentences loaded." % len(self.sentences))

        if vocab is None:
            # If the vocabulary is not provided, calcuate it from data.
            self.vocab = {
                "words": Vocab(word_cnt, specials=[PAD, UNKNOWN]),
                "tags": Vocab(tag_cnt, specials=[]),
            }
        else:
            # Otherwise, reuse the existing vocabulary.
            self.vocab = vocab
        self.unknown_idx = self.vocab["words"].stoi[UNKNOWN]
        self.pad_idx = self.vocab["words"].stoi[PAD]

    def __getitem__(self, idx: int) -> Sentence:
        """
        Get the idx'th sentence in the dataset.
        """
        return self.sentences[idx]

    def __len__(self) -> int:
        """
        Return the number of sentences in the dataset.
        """
        # TODO: Implement this method
        # START HERE
        raise NotImplementedError
        # END

    def form_batch(self, sentences: List[Sentence]) -> Dict[str, Any]:
        """
        A customized function for batching a number of sentences together.
        Different sentences have different lengths. Let max_len be the longest length.
        When packing them into one tensor, we need to pad all sentences to max_len. 
        Return values:
            `words`: a list in which each element itself is a list of words in a sentence
            `word_idxs`: a batch_size x max_len tensor. 
                       word_idxs[i][j] is the index of the j'th word in the i'th sentence .
            `tags`: a list in which each element itself is a list of tags in a sentence
            `tag_idxs`: a batch_size x max_len tensor
                      tag_idxs[i][j] is the index of the j'th tag in the i'th sentence.
            `valid_mask`: a batch_size x max_len tensor
                        valid_mask[i][j] is True if the i'th sentence has the j'th word.
                        Otherwise, valid[i][j] is False.
        """
        words: List[List[str]] = []
        tags: List[List[str]] = []
        max_len = -1  # length of the longest sentence
        for sent in sentences:
            words.append([])
            tags.append([])
            for w, t in sent:
                words[-1].append(w)
                tags[-1].append(t)
            max_len = max(max_len, len(words[-1]))

        batch_size = len(sentences)
        word_idxs = torch.full(
            (batch_size, max_len), fill_value=self.pad_idx, dtype=torch.int64
        )
        tag_idxs = torch.full_like(word_idxs, fill_value=self.vocab["tags"].stoi["O"])
        valid_mask = torch.zeros_like(word_idxs, dtype=torch.bool)

        ## TODO: Fill in the values in word_idxs, tag_idxs, and valid_mask
        ## Caveat: There may be out-of-vocabulary words in validation data
        ## See torchtext.vocab.Vocab: https://pytorch.org/text/stable/vocab.html#torchtext.vocab.Vocab
        ## START HERE
        raise NotImplementedError
        # END

        return {
            "words": words,
            "word_idxs": word_idxs,
            "tags": tags,
            "tag_idxs": tag_idxs,
            "valid_mask": valid_mask,
        }


def create_sequence_dataloaders(
    batch_size: int, shuffle: bool = True
) -> Tuple[DataLoader, DataLoader, Vocab]:
    """
    Create the dataloaders for training and validaiton.
    """
    ds_train = SequenceDataset("eng.train")
    ds_val = SequenceDataset("eng.val", vocab=ds_train.vocab)
    loader_train = DataLoader(
        ds_train,
        batch_size,
        shuffle,
        collate_fn=ds_train.form_batch,  # customized function for batching
        drop_last=True,
        pin_memory=True,
    )
    loader_val = DataLoader(
        ds_val, batch_size, collate_fn=ds_val.form_batch, pin_memory=True
    )
    return loader_train, loader_val, ds_train.vocab