# for loading custom code and packages
install.packages("devtools")
install.packages("remotes")

# utilities
## data processing
install.packages("dplyr")
install.packages("tidyr")
install.packages("tidyverse")
install.packages("magrittr")

## data loading
install.packages("xlsx")
install.packages("readxl")
install.packages("pdftools")
install.packages("rjson")

## plotting and visualization
install.packages("ggplot2")
install.packages("wordcloud")
install.packages("plotly")

## in case you get java related errors:
install.packages("rJava")

# for nlp
install.packages("tidytext") # text processing
install.packages("corpus") # corpus processing
install.packages("openNLP") # general nlp
install.packages("NLP") # general nlp
install.packages("quanteda")    #lsa, text processing
install.packages("topicmodels") #lda
install.packages("textreuse")    #lsh
install.packages("text2vec") # GloVe
install.packages("tidytext") # text processing
install.packages("spacyr") # named entity recognition
install.packages("tm") # topic modelling


# custom nlp
remotes::install_github("bmschmidt/wordVectors") #w2V
install.packages("keras") #deep neural networks


require(devtools)
# corpora data
devtools::install_github("iqss/dataverse-client-r") #dataverse corpora
devtools::install_github("quanteda/quanteda.corpora") # quanteda corpora

