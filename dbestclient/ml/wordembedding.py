import logging
import multiprocessing

import numpy as np

# import pandas as pd
from gensim.models import FastText

# from numpy.core.defchararray import startswith

# https://towardsdatascience.com/word-embedding-with-FastText-and-fasttext-a209c1d3e12c

logger = logging.getLogger(__name__)


class SkipGram:
    def __init__(self):
        # self.embedding = None
        self.dim = None
        self.usecols = None
        self.embeddings = {}
        self.header_categorical = None

    def fit(
        self,
        # gb_data,
        # equal_data,
        categorical_data,
        range_data,
        label_data=None,
        usecols=None,
        dim=20,
        window=3,
        min_count=0,
        negative=30,
        iters=30,
        workers=-1,
        NG=1,
        b_reg=True,
    ):
        ################################################################################
        # This code may cause errors due to dimensionality mismatch.
        categoricals = (
            np.concatenate(
                (categorical_data, range_data[:, np.newaxis].astype(str)), axis=1
            )
            if range_data is not None
            else categorical_data
        )
        ################################################################################

        if b_reg:
            ############################################################################
            # This code may causes errors due to dimensionality mismatch.
            categoricals = (
                np.concatenate(
                    (categoricals, label_data[:, np.newaxis].astype(str)), axis=1
                )
                if label_data is not None
                else categorical_data
            )
            ############################################################################

        if usecols is not None:
            self.usecols = usecols
            # print("self.usecols", self.usecols)
            header = (
                [
                    usecols["gb"]
                    + usecols["x_categorical"]
                    + usecols["x_continous"]
                    + [usecols["y"][0]]
                ]
                if b_reg
                else [usecols["gb"] + usecols["x_categorical"] + usecols["x_continous"]]
            )
            self.header_categorical = (
                usecols["gb"] + usecols["x_categorical"]
                if usecols["x_categorical"] is not None
                else usecols["gb"]
            )
        else:
            tmp = [
                "aa_",
                "bb_",
                "cc_",
                "dd_",
                "ee_",
                "ff_",
                "gg_",
                "hh_",
                "ii_",
                "jj_",
                "kk_",
                "ll_",
                "mm_",
                "nn_",
            ]
            self.header_categorical = tmp[: len(categorical_data[0])]
            # print(categoricals, "---->>>>>")
            # print(categoricals[0])
            # print(len(categoricals[0]))
            # header = tmp[:3]
            header = [tmp[: len(categoricals[0])]]

        # print(categoricals)
        # print(header)
        headers = np.repeat(header, len(categoricals), axis=0)
        # print("*" * 70)
        # print(headers, "-" * 20, ">" * 20)
        NG = len(self.header_categorical)
        self.dim = dim * len(self.header_categorical)

        ################################################################################
        # Convert both arrays to string to prevent numpy type errors
        sentences = np.core.defchararray.add(
            headers.astype(str), categoricals.astype(str)
        ).tolist()
        ################################################################################
        # print(sentences)

        workers = multiprocessing.cpu_count() if workers == -1 else 1
        model = FastText(
            sentences,
            vector_size=int(self.dim / NG),
            window=window,
            min_count=min_count,
            negative=negative,
            epochs=iters,
            workers=workers,
        )

        # word_vectors = model.wv  # Matix of model
        vocab = model.wv.key_to_index  # Vocabulary
        #self.dim = dim * len(self.header_categorical)
        # print("dim is", self.dim)
        # print(model["citylondon"])

        for word in vocab:
            # print("word", word)
            for head in self.header_categorical:
                # print("head", head)
                if word.startswith(head):
                    self.embeddings[word] = model.wv[word]
        # print(self.embeddings.keys())
        logger.debug("Finished training embedding.")
        return self

    def predicts(self, keys):
        # print("keys,", type(keys))
        # print("keys,", keys)
        headers = np.repeat([self.header_categorical], len(keys), axis=0)
        # print("headers", headers)
        sentences = np.core.defchararray.add(headers.astype(str), keys.astype(str))
        # print("sentences",sentences)
        # print("self.embeddings", self.embeddings.keys())

        # exit()

        col0 = sentences[:, 0]
        predictions = np.array([self.embeddings[i] for i in col0])
        # print("first columns ", predictions)

        for col_idx in range(1, len(sentences[0])):
            col = [x.replace(";", "") for x in sentences[:, col_idx]]
            prediction_col = np.array([self.embeddings[i] for i in col])
            predictions = np.concatenate((predictions, prediction_col), axis=1)
        return predictions

        # print("predictions are ")
        # print(predictions)

        # print("column 1")
        # print("^"*40)
        # for i in col0:
        #     print(self.embeddings[i])
        # print("^"*40)

        # predictions = None
        # for words in sentences:
        #     # print(words)
        #     prediction = np.array([])
        #     for word in words:
        #         # print(self.embeddings[word])
        #         prediction = np.append(prediction, self.embeddings[word])[
        #             np.newaxis, :
        #         ]  # .tolist()
        #         # print("prediction")
        #         # print(prediction)
        #     predictions = (
        #         np.append(predictions, prediction, axis=0)
        #         if predictions is not None
        #         else prediction
        #     )
        # # exit()
        # print(predictions)
        # return predictions
        # print(keys)

    def predicts_low_efficient(self, keys):
        headers = np.repeat([self.header_categorical], len(keys), axis=0)
        # print(headers)
        sentences = np.core.defchararray.add(headers, keys)  # .tolist()
        # print(sentences)
        # print("self.embeddings", self.embeddings.keys())

        predictions = None
        for words in sentences:
            # print(words)
            prediction = np.array([])
            for word in words:
                # print(self.embeddings[word])
                prediction = np.append(prediction, self.embeddings[word])[
                    np.newaxis, :
                ]  # .tolist()
                # print("prediction")
                # print(prediction)
            predictions = (
                np.append(predictions, prediction, axis=0)
                if predictions is not None
                else prediction
            )
        return predictions
        # print(keys)
