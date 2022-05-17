# Overview
- Graph-Neural-Networks (GNN) represents a given graph where the nodes are linked with edges into a vector space. Relational-Graph-Convolutional-Networks (R-GCN) introduced relation dependant graph representation. The typed dependency of given text forms triples of words and dependency (t<sub>dep</sub>, dependency, t<sub>gov</sub>) is similar to the triples of nodes and edge in a graph (d<sub>i</sub>, rel, d<sub>j</sub>) where t is token and d is data. Graph Attention Networks (GAT) represents a given graph leveraging masked self-attention layer. This project aims to implement the R-GCN and GAT for span detection from given text.

# Brief description
- text_processing.py
> Output format
> - output: Tokenized result of a given text. (list)
- my_onehot.py
> Output format
> - output: List of tensor of input tokens. (Tensor)
- depgcn.py; depgat.py
> Output format
> - output: List of tensor of the sum of token embedding itself and dependency relation that is connected to the governor. (Tensor)

# Prerequisites
- argparse
- torch
- stanza
- spacy
- nltk
- gensim
- tqdm

# Parameters
- nlp_pipeline(str, defaults to "stanza"): NLP preprocessing pipeline.
- unk_ignore(bool, defaults to True): Ignore unseen word or not.
- num_layers(int, defaults to 1): The number of lstm/bilstm layers.
- gnn(str, defaults to "depgcn"): Type of dependency gnn layer. (depgcn, depgat)
- alpha(float, defaults to 0.01): Controls the angle of the negative slope of LeakyReLU.
- epochs(int, defaults to 100): The number of epochs for training.
- learning_rate(float, defaults to 1e-2): Learning rate.
- dropout_rate(float, defaults to 0.1): Dropout rate.
- reverse(bool, defaults to True): Applying reverse dependency cases (gov -> dep) or not.

# References
- Graph Convolutional Networks (GCN): Kipf, T. N., & Welling, M. (2016). Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907.
- Relational-Graph-Convolutional-Networks (R-GCN): Schlichtkrull, M., Kipf, T. N., Bloem, P., Berg, R. V. D., Titov, I., & Welling, M. (2018, June). Modeling relational data with graph convolutional networks. In European semantic web conference (pp. 593-607). Springer, Cham.
- Attention: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.
- Graph Attention Networks (GAT): Veličković, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y. (2017). Graph attention networks. arXiv preprint arXiv:1710.10903.
- Stanza: Qi, P., Zhang, Y., Zhang, Y., Bolton, J., & Manning, C. D. (2020). Stanza: A Python natural language processing toolkit for many human languages. arXiv preprint arXiv:2003.07082.
- Spacy: Matthew Honnibal and Ines Montani. 2017. spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing. To appear (2017).
- NLTK: Bird, Steven, Edward Loper and Ewan Klein (2009). Natural Language Processing with Python. O'Reilly Media Inc.
- Gensim: Rehurek, R., & Sojka, P. (2010). Software framework for topic modelling with large corpora. In In Proceedings of the LREC 2010 workshop on new challenges for NLP frameworks.
- Sample data: Yang, X., Obadinma, S., Zhao, H., Zhang, Q., Matwin, S., & Zhu, X. (2020). SemEval-2020 task 5: Counterfactual recognition. arXiv preprint arXiv:2008.00563.
