from gensim import models
from gensim.models import CoherenceModel


class GridSearch:

    def __init__(self, corpus, id2word, param_dictionary):
        """
        Execute LDA,LSI,HDP models
        :param corpus: Stream of document vectors  in BoW format
        :param id2word: Mapping of word and word ids
        :param param_dictionary: dictionary of single key, pair values that contains additional parameters
        """
        self.corpus = corpus
        self.id2word = id2word
        self.param_dictionary = param_dictionary

    def nlp_model(self, model):
        """
        Execute lda, lsi, hdp models
        :param model: specify LDA,LSI, OR HDP model
        :return: gensim model
        """
        if model == 'lda-multicore':
            return models.LdaMulticore(self.corpus, id2word=self.id2word, **self.param_dictionary)
        elif model == 'lda':
            return models.LdaModel(self.corpus, id2word=self.id2word, **self.param_dictionary)
        elif model == 'lsi':
            return models.LsiModel(self.corpus, id2word=self.id2word, **self.param_dictionary)
        elif model == 'hdp':
            return models.HdpModel(self.corpus, id2word=self.id2word, **self.param_dictionary)
        else:
            return 'Sorry, we only support HDP, LDA, LSI models'

    def get_coherence_score(self, model, text, coherence_metric):
        """
        :param text:
        :param model: specify LDA,LSI, or HDP model
        :param coherence_metric: coherence metric we want to evaluate.
        :return: coherence score
        """
        coherence_models = CoherenceModel(model=GridSearch.nlp_model(self, model),
                                          texts=text, dictionary=self.id2word,
                                          coherence=coherence_metric)  # type: CoherenceModel
        return coherence_models.get_coherence()
