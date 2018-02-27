# attr2vec


TensorFlow implementation of the attr2vec model, based on the following publication:

- Fabio Petroni, Vassilis Plachouras, Timothy Nugent and Jochen L. Leidner: "attr2vec: : Jointly Learning Word and Contextual Attribute Embeddings with Factorization Machines." In: Proceedings of the 16th Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL), 2018.

If you use the application please cite the paper.



### Modeling input data

The input corpus is represented as two files: *Cooccur.csv* and *Word2Id.csv*. 
The first file follows the original libfm format (http://www.libfm.org) and contains the target vector **Y** as well as the feature matrix **X**.


We will use an example to concretely show how to model the input data, using as corpus the following text:

```
Prime Minister Theresa May will remind her cabinet that discussions must remain private. 
Theresa Mary May is a British politician who has served as Prime Minister.
```

The folder *data_pos* contains the modeling of such example corpus using Part-of-Speech (POS) as additional contextual attribute, while the folder *data_dependency* contains the input data to train dependency-based embeddings.

The *Word2Id.csv* file contains the symbols vocabulary, and looks like this:
```
"IN",2,2
"NNP",6,2
"NNS",7,2
"discussions",17,0
"minister",23,0
"prime",26,0
"that",31,0
[...]
```
The first column contains the word form or the POS tag, the second column an unique identifier, the third column a meta information to distinguish words from POS tags (i.e., 0 for words, 2 for POS tag).


The *Cooccur.csv* file looks like this:

```
1.0 17:1.0 31:1.0 7:1.0 2:1.0
2.0 23:1.0 26:1.0 6:2.0
[...]
```
Please read the libfm manual (http://www.libfm.org/libfm-1.42.manual.pdf) for an extensive description of this format. Here, the first line conveys the information that symbols with id 17, 31, 7, 2 (all with value 1.0) co-occur in the corpus with frequency 1.0.

### Train the attr2vec model

To train the attr2vec model on the example data simply run
```{r, engine='bash', count_lines}
python train.py
```
Open the file and edit it to change the paramenters.

The application will write vectors and model metadata in the *log* folder.
You can use TensorBoard to explore the model internals, as follows:

```{r, engine='bash', count_lines}
tensorboard --logdir log/
```

![TensorBoard](screenshots/tensorboard2.png?raw=true "TensorBoard")