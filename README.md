# text_embeddings

## Intro
A repository containing the material for the "Representation Learning and Text Embeddings" lecture for the PoliTics expert workshop.
Material regarding the workshop will be continuously uploaded before the workshop.

## Workshop Details 

*When*: 19 and 20 November 2019, 5-8:30 pm<br/>
*Where*: room AFL-H-372, Department of Political Science, Affolternstrasse 56, 8050 Zurich<br/>
~~*Registration*: closed~~ 

## Readings

### In political science 

The following entries can be considered core papers in political science, as they explain word embedding methods in a simplified way and, more importantly, showcase their application to problems relevant to a political science audience.

#### Overview articles 

- Arthur Spirling and Pedro L. Rodriguez (2019) "Word Embeddings: What works, what doesn’t, and how to tell the difference for applied research" [PDF](https://www.nyu.edu/projects/spirling/documents/embed.pdf)

#### Applications to substantive questions

- Ludovic Rheault and Christopher Cochrane (2019) "Word Embeddings for the Analysis of Ideological Placement in Parliamentary Corpora" [DOI](https://doi.org/10.1017/pan.2019.26)
- Stefano Gurciullo and Slava Mikhaylov (2017) "Detecting Policy Preferences and Dynamics in the UN General Debate with Neural Word Embeddings" [URL](https://arxiv.org/abs/1707.03490)
- Emma Rodman (2019) "A Timely Intervention: Tracking the Changing Meanings of Political Concepts with Word Vectors" [DOI](https://doi.org/10.1017/pan.2019.23)

### Foundational papers

Some of the foundational (and more technical) papers listed below may also help you to to deepen your understanding of language models in representation learning.
In the workshop, we will focus on attaining a high-level intuition of these different methods, and we'll discuss their different uses for specific use-cases, e.g. why would I want to pick a sequential model on tweets but an attention model on paragraphs.

#### An overview on "Representation Learning":
- Bengio, Y., Courville, A., & Vincent, P. (2013). Representation learning: A review and new perspectives. IEEE transactions on pattern analysis and machine intelligence, 35(8), 1798-1828. [PDF](https://arxiv.org/pdf/1206.5538.pdf)

#### NLP tasks:
- Hashimoto, K., Xiong, C., Tsuruoka, Y., & Socher, R. (2016). A joint many-task model: Growing a neural network for multiple nlp tasks. arXiv preprint arXiv:1611.01587. [PDF](https://arxiv.org/pdf/1611.01587.pdf)
- Hotho, A., Nürnberger, A., & Paaß, G. (2005, May). A brief survey of text mining. In Ldv Forum (Vol. 20, No. 1, pp. 19-62).[PDF](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.447.4161&rep=rep1&type=pdf)
- Allahyari, M., Pouriyeh, S., Assefi, M., Safaei, S., Trippe, E. D., Gutierrez, J. B., & Kochut, K. (2017). A brief survey of text mining: Classification, clustering and extraction techniques. arXiv preprint arXiv:1707.02919. [PDF](https://arxiv.org/pdf/1707.02919.pdf) 

#### Preprocessing:
- Haddi, E., Liu, X., & Shi, Y. (2013). The role of text pre-processing in sentiment analysis. Procedia Computer Science, 17, 26-32.[URL](https://www.sciencedirect.com/science/article/pii/S1877050913001385)
- Jianqiang, Z., & Xiaolin, G. (2017). Comparison research on text pre-processing methods on twitter sentiment analysis. IEEE Access, 5, 2870-2879.[URL](https://ieeexplore.ieee.org/abstract/document/7862202/)
- Vijayarani, S., Ilamathi, M. J., & Nithya, M. (2015). Preprocessing techniques for text mining-an overview. International Journal of Computer Science & Communication Networks, 5(1), 7-16.[PDF](https://pdfs.semanticscholar.org/1fa1/1c4de09b86a05062127c68a7662e3ba53251.pdf)

#### Older methods
- Ramos, J. (2003, December). Using tf-idf to determine word relevance in document queries. In Proceedings of the first instructional conference on machine learning (Vol. 242, pp. 133-142). [PDF](https://www.cs.rutgers.edu/~mlittman/courses/ml03/iCML03/papers/ramos.pdf)
- Wu, H. C., Luk, R. W. P., Wong, K. F., & Kwok, K. L. (2008). Interpreting tf-idf term weights as making relevance decisions. ACM Transactions on Information Systems (TOIS), 26(3), 13. [URL](https://dl.acm.org/citation.cfm?id=1361686)
- Jacobi, C., Van Atteveldt, W., & Welbers, K. (2016). Quantitative analysis of large amounts of journalistic texts using topic modelling. Digital Journalism, 4(1), 89-106. [PDF](https://www.researchgate.net/profile/Wouter_Atteveldt/publication/283671339_Quantitative_analysis_of_large_amounts_of_journalistic_texts_using_topic_modelling/links/5645b19d08ae451880a9b8a0.pdf)
- Nikolenko, S. I., Koltcov, S., & Koltsova, O. (2017). Topic modelling for qualitative studies. Journal of Information Science, 43(1), 88-102.[URL](https://journals.sagepub.com/doi/abs/10.1177/0165551515617393?journalCode=jisb)
- Uys, J. W., Du Preez, N. D., & Uys, E. W. (2008, July). Leveraging unstructured information using topic modelling. In PICMET'08-2008 Portland International Conference on Management of Engineering & Technology (pp. 955-961). IEEE.[URL](https://ieeexplore.ieee.org/abstract/document/4599703/)
- Ravichandran, D., Pantel, P., & Hovy, E. (2005, June). Randomized algorithms and NLP: Using locality sensitive hash functions for high speed noun clustering. In Proceedings of the 43rd Annual Meeting of the Association for Computational Linguistics (ACL’05) (pp. 622-629). [PDF](https://www.aclweb.org/anthology/P05-1077.pdf)

#### Word embeddings (basics):
- Stratos, K., Collins, M., & Hsu, D. (2015, July). Model-based word embeddings from decompositions of count matrices. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers) (pp. 1282-1291).[URL](https://www.aclweb.org/anthology/P15-1124/)
- Rong, X. (2014). word2vec parameter learning explained. arXiv preprint arXiv:1411.2738.[PDF](https://arxiv.org/pdf/1411.2738.pdf)
- Goldberg, Y., & Levy, O. (2014). word2vec Explained: deriving Mikolov et al.'s negative-sampling word-embedding method. arXiv preprint. [PDF](https://arxiv.org/abs/1402.3722)
- Shi, T., & Liu, Z. (2014). Linking GloVe with word2vec. arXiv preprint arXiv:1411.5595. [PDF](https://arxiv.org/abs/1411.5595.pdf)
- Pennington, J., Socher, R., & Manning, C. (2014, October). Glove: Global vectors for word representation. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP) (pp. 1532-1543). [URL](https://www.aclweb.org/anthology/D14-1162/)
- Levy, O., & Goldberg, Y. (2014, June). Dependency-based word embeddings. In Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers) (pp. 302-308).[PDF](https://www.aclweb.org/anthology/P14-2050.pdf)
- Liu, Y., Liu, Z., Chua, T. S., & Sun, M. (2015, February). Topical word embeddings. In Twenty-Ninth AAAI Conference on Artificial Intelligence.[URL](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/viewPaper/9314)
- Mandelbaum, A., & Shalev, A. (2016). Word embeddings and their use in sentence classification tasks. arXiv preprint arXiv:1610.08229.[PDF](https://arxiv.org/abs/1610.08229.pdf)

#### Document, paragraph embeddings:
- Kusner, M., Sun, Y., Kolkin, N., & Weinberger, K. (2015, June). From word embeddings to document distances. In International conference on machine learning (pp. 957-966).[PDF](http://proceedings.mlr.press/v37/kusnerb15.pdf)
- Lau, J. H., & Baldwin, T. (2016). An empirical evaluation of doc2vec with practical insights into document embedding generation. arXiv preprint arXiv:1607.05368.[PDF](https://arxiv.org/pdf/1607.05368.pdf)
- Dai, A. M., Olah, C., & Le, Q. V. (2015). Document embedding with paragraph vectors. arXiv preprint arXiv:1507.07998. [PDF](https://arxiv.org/pdf/1507.07998)

#### Word embedding evaluation:
- Schnabel, T., Labutov, I., Mimno, D., & Joachims, T. (2015). Evaluation methods for unsupervised word embeddings. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (pp. 298-307).[URL](https://www.aclweb.org/anthology/D15-1036)
- Faruqui, M., Tsvetkov, Y., Rastogi, P., & Dyer, C. (2016). Problems with evaluation of word embeddings using word similarity tasks. arXiv preprint arXiv:1605.02276.[PDF](https://arxiv.org/abs/1605.02276.pdf)
- Iacobacci, I., Pilehvar, M. T., & Navigli, R. (2016, August). Embeddings for word sense disambiguation: An evaluation study. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) (pp. 897-907).[URL](https://www.aclweb.org/anthology/P16-1085)
- Lau, J. H., & Baldwin, T. (2016). An empirical evaluation of doc2vec with practical insights into document embedding generation. arXiv preprint arXiv:1607.05368.[PDF](https://arxiv.org/abs/1607.05368.pdf)
- Ghannay, S., Favre, B., Esteve, Y., & Camelin, N. (2016, May). Word embedding evaluation and combination. In Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC'16) (pp. 300-305).[PDF](http://www.lrec-conf.org/proceedings/lrec2016/pdf/392_Paper.pdf)
- Zuccon, G., Koopman, B., Bruza, P., & Azzopardi, L. (2015, December). Integrating and evaluating neural word embeddings in information retrieval. In Proceedings of the 20th Australasian document computing symposium (p. 12). ACM. [PDF](https://dl.acm.org/citation.cfm?id=2838936)
- Arora, S., Liang, Y., & Ma, T. (2016). A simple but tough-to-beat baseline for sentence embeddings.[PDF](https://openreview.net/pdf?id=SyK00v5xx)

### Advanced readings and state-of-the-art:
Several of the methods below require a deeper understanding of deep learning.
Furthermore, although these models are extremely powerful, they require a lot of computational resources and data to be effective.
Still, many of those models are (or will be) implemented in widely used NLP packages.
The relevant papers below and especially the parts discussing the intuition behind the design of such models can support in deciding when it is worth the effort to use them.

#### Recurrent networks (words as sequence):
- Dai, A. M., & Le, Q. V. (2015). Semi-supervised sequence learning. In Advances in neural information processing systems (pp. 3079-3087). [URL](http://papers.nips.cc/paper/5949-semi-supervised-sequence-learning)
- Kiros, R., Zhu, Y., Salakhutdinov, R. R., Zemel, R., Urtasun, R., Torralba, A., & Fidler, S. (2015). Skip-thought vectors. In Advances in neural information processing systems (pp. 3294-3302).[URL](http://papers.nips.cc/paper/5950-skip-thought-vectors)
- Mueller, J., & Thyagarajan, A. (2016, March). Siamese recurrent architectures for learning sentence similarity. In Thirtieth AAAI Conference on Artificial Intelligence.[URL](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewPaper/12195)
- Clark, C., & Gardner, M. (2017). Simple and effective multi-paragraph reading comprehension. arXiv preprint arXiv:1710.10723.[PDF](https://openreview.net/pdf?id=SyK00v5xx)

#### Recursive and topological (tree-like):
- Gehring, J., Auli, M., Grangier, D., Yarats, D., & Dauphin, Y. N. (2017, August). Convolutional sequence to sequence learning. In Proceedings of the 34th International Conference on Machine Learning-Volume 70 (pp. 1243-1252). JMLR. org.[URL](https://dl.acm.org/citation.cfm?id=3305510)
- Socher, R., Perelygin, A., Wu, J., Chuang, J., Manning, C. D., Ng, A., & Potts, C. (2013, October). Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the 2013 conference on empirical methods in natural language processing (pp. 1631-1642). [PDF](https://www-nlp.stanford.edu/pubs/SocherEtAl_EMNLP2013.pdf)
- Dos Santos, C., & Gatti, M. (2014, August). Deep convolutional neural networks for sentiment analysis of short texts. In Proceedings of COLING 2014, the 25th International Conference on Computational Linguistics: Technical Papers (pp. 69-78). [PDF](https://www.aclweb.org/anthology/C14-1008)
- Lai, S., Xu, L., Liu, K., & Zhao, J. (2015, February). Recurrent convolutional neural networks for text classification. In Twenty-ninth AA. [PDF](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9745/9552)

#### Attention based (graph based):
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. In Advances in neural information processing systems (pp. 5998-6008). [URL](http://papers.nips.cc/paper/7181-attention-is-all-you-need)
- Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. [PDF](arXiv preprint arXiv:1810.04805.pdf)


### Books

Complete readings. Very useful for going in depth.

- Bender, E. M. (2013). Linguistic fundamentals for natural language processing: 100 essentials from morphology and syntax. Synthesis lectures on human language technologies, 6(3), 1-184.[URL](https://www.morganclaypool.com/doi/abs/10.2200/S00493ED1V01Y201303HLT020)
- Parsing, C. (2009). Speech and language processing.[URL](https://web.stanford.edu/~jurafsky/slp3/edbook_oct162019.pdf)
- Manning, C. D., Manning, C. D., & Schütze, H. (1999). Foundations of statistical natural language processing. MIT press.[URL](https://nlp.stanford.edu/fsnlp/)
- Schütze, H., Manning, C. D., & Raghavan, P. (2008, June). Introduction to information retrieval. In Proceedings of the international communication of association for computing machinery conference (p. 260).[PDF](https://www2.kbs.uni-hannover.de/fileadmin/institut/pdf/ti1/slides/08eval.pdf)

### Websites

A bit less scientific and more intuitive presentation of the concepts above.


#### General:
- [A General Approach to Preprocessing Text Data](
https://www.kdnuggets.com/2017/12/general-approach-preprocessing-text-data.html)
- [The Main Approaches to Natural Language Processing Tasks](https://www.kdnuggets.com/2018/10/main-approaches-natural-language-processing-tasks.html)
- [List of natural language processing tasks
](https://natural-language-understanding.fandom.com/wiki/List_of_natural_language_processing_tasks)
- [Stanford: CS224n: Natural Language Processing with Deep Learning](https://www.youtube.com/playlist?list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z)
https://towardsdatascience.com/nlp-extracting-the-main-topics-from-your-dataset-using-lda-in-minutes-21486f5aa925
- [YSDA Natural Language Processing course* on Github:](https://github.com/yandexdataschool/nlp_course)

#### Older methods:
- [Bag of Words and Tf-idf Explained](http://datameetsmedia.com/bag-of-words-tf-idf-explained/)
- [LDA](https://towardsdatascience.com/nlp-extracting-the-main-topics-from-your-dataset-using-lda-in-minutes-21486f5aa925)
- [Locality Sensitive Hashing - Towards Data Science](
https://towardsdatascience.com/understanding-locality-sensitive-hashing-49f6d1f6134)

#### Word embeddings:
- [Introduction to Word Embedding and Word2Vec](https://towardsdatascience.com/introduction-to-word-embedding-and-word2vec-652d0c2060fa)
- [What are word embeddings](https://machinelearningmastery.com/what-are-word-embeddings/)
- [An Intuitive Understanding of Word Embeddings: From Count Vectors to Word2Vec](https://www.analyticsvidhya.com/blog/2017/06/word-embeddings-count-word2veec/)
- [Deep	Learning	for	NLP (without	Magic)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.400.1058&rep=rep1&type=pdf)
- [The Current Best of Universal Word Embeddings and Sentence Embeddings](https://medium.com/huggingface/universal-word-sentence-embeddings-ce48ddc8fc3a)
- [BERT Explained: State of the art language model for NLP](https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270)
- [Embeddings](https://blogs.rstudio.com/tensorflow/posts/2017-12-22-word-embeddings-with-keras/)
- [The Transformer – Attention is all you need.](https://mchromiak.github.io/articles/2017/Sep/12/Transformer-Attention-is-all-you-need/)
- [Word Embeddings for Sentence Classification](https://towardsdatascience.com/word-embeddings-for-sentence-classification-c8cb664c5029)

#### Political science:
- [Analyzing the Potential of Machine Learning in Political Science](https://medium.com/@mandlahsibanda/analyzing-the-potential-of-machine-learning-in-political-science-f1136dc1d2c)
- [Political Dashboard](https://political-dashboard.com/)
- [Tweet Like Trump](https://towardsdatascience.com/tweet-like-trump-with-a-one2seq-model-cb1461f9d54c)
- [How Trump Reshaped the Presidency in Over 11,000 Tweets](https://www.nytimes.com/interactive/2019/11/02/us/politics/trump-twitter-presidency.html)

### Code tutorials
Some code repositories and examples mostly with R and word embeddings:
- [Text preprocessing](https://quanteda.io/)
- [LDA](https://www.tidytextmining.com/topicmodeling.html)
- [LSA](https://quanteda.io/articles/pkgdown/examples/lsa.html)
- [LSH](https://rdrr.io/cran/textreuse/man/lsh.html)
- [GloVe](http://text2vec.org/vectorization.html)
- [Word2Vec](https://rdrr.io/github/bmschmidt/wordVectors/)
- [Bert](https://blogs.rstudio.com/tensorflow/posts/2019-09-30-bert-r/)
- [Keras](https://keras.rstudio.com/)
