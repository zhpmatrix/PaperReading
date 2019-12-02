### Others

24.《A Deep Look into Neural Ranking Models for Information Retrieval》

一篇关于Ranking的综述文章。hinge loss可以用在pairwise的ltr任务中。

23.《PyTorch: An Imperative Style, High-Performance Deep Learning Library》，NIPS2019

讨论系统哲学的学术文章，其实官方博客已经讨论了很多相关思想了，所以这篇作为学术文章可能就是方便大家引用吧。作为宇宙最屌框架，PyTorch已经不需要用一篇文章的引用量来证明自己了。

22.《Using Knowledge Graphs for Fact-Aware Language Modeling》，ACL2019

相关的想法：

（1）bert系通过mlm能否直接fact-aware？已经有人做了一些探索。

（2）没有没关系。把kg的信息融入encoder端，提升decoder的能力，就是这篇了。

（3）结合各个任务，融合kg的信息，那么怎么融合kg的信息，就是一个可以玩儿出花儿来的事情。

21.《Spam Review Detection with Graph Convolutional Networks》，CIKM2019最佳应用论文

将GCN用于闲鱼的垃圾营销广告检测。特征提取层上分为两个部分，分别是基于商品-用户-评论的二部图和不同评论构成的图，GCN作为特征提取器。分类层采用TextCNN。

20.《Chinese Word Segmentation as Character Tagging》，2003年的文章，引用量很高。

好像是第一次将分词问题modeling为一个tagging问题来做，共有四个标签（LL/MM/RR/LR）如下：

![img_word_segmentation](https://wx3.sinaimg.cn/mw690/aba7d18bly1g9cvr16rhwj20t808t40d.jpg)

19.《How to Ask Better Questions? A Large-Scale Multi-Domain Dataset for Rewriting Ill-Formed Questions》

做了一个问题改写的数据集。

18.《Learning Semantic Hierarchies via Word Embeddings》，ACL2014

《大词林》，用word2vec的思路做“上下位”关系挖掘。word2vec用好了，应该可以解决很多问题，包括embedding，相关性等任务。

用于开放域的实体类别层次化（上下位关系）。核心思想是用词向量之间的差值刻画上下位关系。工作分为三个步骤：词向量学习，映射矩阵学习和上下位关系判断。上下位关系的判断也可归于关系抽取任务，是一种特殊的较为抽象的关系。因此可以很自然地将文章的想法拓展到关系分类任务上，对每类关系学习一个映射矩阵。当用于SPO三元组直接抽取时，需要有针对性的映射关系学习方法。

17.《Statistical Machine Translation: IBM Models 1 and 2》，Michael Collins

Noisy Channel Model的经典案例：作为统计模型用于机器翻译任务。除此之外，可以用于拼写纠错，Auto Suggestion等，大二时实现的Bayes Matting也有类似的感觉。总之，NCM是一个比较general的理论模型。

16.《Few-Shot Sequence Labeling with Label Dependency Transfer and Pair-wise Embedding》

 Few-Shot Learning的工作，用于命令实体识别任务上。

15.《Hierarchically-Reﬁned Label Attention Network for Sequence Labeling》,EMNLP2019

Label Embedding+Attention用于sequence labeling。平均提升了不足一个百分点，可能和相关任务原来的指标已经较高有关系。不过工作做的还是比较干净的。

14.《Multi-instance Multi-label Learning for Relation Extraction》

2012年的工作，引用量400+。将distant-supervision得到的数据，建模为一个miml问题。主要技术：graphical model+EM。文章引用了周志华老师的miml的相关工作。

延伸思考：

（1）snorkel的相关技术可以结合distant-supervision，比如denoising。

（2）更加general的relation extraction可以关注fewrel的进展。

（3）distant-supervision需要关注multi-instance的相关工作。

相关工作：《Relation Extraction with Multi-instance Multi-label Convolutional Neural Networks》

13.《Distant supervision for relation extraction without labeled data》

2009年的工作，引用非常多，但是并非distant-supervision第一次提出，作者之一是Dan Jurafsky。文章在关系抽取的时候，还是基于人工构建的feature，包括syntactic和lexicial feature等。

相关启发：

（1）regular expr是learnable的；instance和pattern的snowball式玩法。

（2）distant-supervision下，multi-instance的利用是亮点。

12.《KG-BERT: BERT for Knowledge Graph Completion》

这篇工作采用了和之前事件判别模型类似的思路，不同之处在于直接基于三元组做，不包含上下文的具体描述，在相关数据集上取得了SOTA结果。虽然作者给出了一些解释，不过文章实验做的不够充足，并且解释似乎不是特别具有说服性。

结论：灌水。

11.**如何使用更多的外部数据提升模型效果？**

（1）《Exploiting Monolingual Data at Scale for Neural Machine Translation》,EMNLP2019

观点：在机翻中，单独地利用src端做Forward Translation和单独地利用tgt端做Back Translation，在模型性能提升到一定阶段后都会有损失，不过二者的损失速度不同。那么，这篇文章综合利用二者提升机翻的效果。

整体上数据利用流程：标注的平行语料->混合语料（标注+伪标签+**加噪**）->伪标签语料。

过程中需要训练/微调的翻译模型个数：8个。

需要强调的是：加噪很重要。

（2）《Self-training with Noisy Student improves ImageNet classification》，两天前的文章

贡献：标准的self-training的范式，将ImageNet的分类指标极限又推进了一点儿。

需要强调的是：student的学习需要加噪。原文如下：

During the generation of the pseudo labels, the teacher is not noised so that the pseudo labels are as good as possible. But during the learning of the student, we inject noise such as data augmentation, dropout, stochastic depth to the stu- dent so that the noised student is forced to learn harder from the pseudo labels. 

上述两篇文章的延伸思考：

  当数据是构造得到的时候，加噪对下游模型的学习可能会是比较重要的一个因素。除了上述两篇文章，类似的观点在ELECTRA的工作中也可以看到是一个非常重要的地方。在一些场景下，数据的构造过程是比较困难的任务。因为，当开始构造数据的时候，其实已经引入了一定的归纳偏置。这里包括：平衡性，难易程度以及与任务强相关的一些信号等。总之，当模型是在自己构建的数据上去learn的时候，对模型在真实数据集上的表现保持警惕应该没有错，本质上还是i.i.d的问题。

10.《ZEN: Pre-training Chinese Encoder Enhanced by N-gram Representations》

号称目前为止最强中文NLP预训练模型。整体思路上可以从deep&wide结构来理解。deep结构和传统的bert类似，wide结构用来encode n-gram的信息。其实，提供了缓解中文bert在词语粒度上modeling不足的问题。

延伸思考：对词语粒度的语义单元modeling是关于中文bert的老问题了。一般有两个思路：第一个思路，直接在deep结构的input端添加embedding信息，但是这样的问题在于可能会引入noise信息；第二个思路，使用wide结构，单独训练词语粒度的embedding，最后在output端进行信息的融合。这里选择了第二种，其实也是一种比较general的思路。

虽然不清楚在哪个粒度建模比较合适，但是从最近的一些工作来看，融合一些高层semantic信息（多粒度的信息）不是一件很坏的事情。不管怎样，整体上是我喜欢的思路，简单有效。

具体图如下：

![img_zen](https://wx4.sinaimg.cn/mw690/aba7d18bly1g8om2b1jsej20iv0gradb.jpg)

9.《ELECTRA: PRE-TRAINING TEXT ENCODERS AS DISCRIMINATORS RATHER THAN GENERATORS》,ICLR2020

主要内容：Thorough experiments demonstrate this new pre-training task is more efﬁ- cient than MLM because the model learns from all input tokens rather than just the small subset that was masked out.

延伸思考：和non-parallel style transfer的工作类似。不过这篇的主要目的是学到一个好的MLM。图示如下：

![img_electra](https://wx3.sinaimg.cn/mw690/aba7d18bly1g8jwz090qcj20ub0dqq6a.jpg)

8.《Pseudolikelihood Reranking with Masked Language Models》

整体上工作类似于一个知识蒸馏，相对简单。如下：

![img_pseudo](https://wx1.sinaimg.cn/mw690/aba7d18bly1g8jqqsumfxj20zd0jgq60.jpg)

7.《Multi-Stage Document Ranking with BERT》

不断地召回和排序。具体如下：

![imag_rank_with_bert](https://wx1.sinaimg.cn/mw690/aba7d18bly1g8jqojc4xjj20t70eead3.jpg)

6.《Class-Balanced Loss Based on Effective Number of Samples》，2019年

损失函数reweight解决imbalanced问题。2009年何海波的一篇综述，做到今天其实还是这些问题，无非换个姿势，再来一次，逃。

5.《Neural Relation Extraction via Inner-Sentence Noise Reduction and Transfer Learning》

神经关系抽取，对input做句法分析，拿到包含两个entity的子树作为context(表示怀疑)，如下：

![img_nre](https://wx1.sinaimg.cn/mw690/aba7d18bly1g8h9smpyevj20p50e6418.jpg)

4.《BPE-Dropout: Simple and Effective Subword Regularization》

很容易想到，有提升。

3.《Language Models as Knowledge Bases》

属于对Bert做Probing的流派，从Bert出来后，相关工作就有几个，颇有盲人摸象的感觉。看别人摸也挺有意思的，这篇的想法与最近做的两个工作有一些联系。

（1）术语纠错。直觉上看是一个Low Resource问题，基本做法是拿Bert在汽车语料上做MLM的fine-tune，句子过来，利用做好的实体识别模型mask掉术语，MLM直接预测mask的术语。初步结果是：有些确实可以预测到正确的术语，多数情况下虽然预测不到正确的术语，但是术语的类型是正确的，比如品牌，厂商类型等，给你的车系取一个霸气的名字就靠MLM啦。

（2）non-parallel的style transfer。和上述对比，直觉上难度要低一些。方法很简单，把pos情感的句子的pos词mask掉，加neg情感的emb，MLM预测mask对应的neg词。mask的位置类型相同，情感相反。这个任务中，mask的位置很重要。实现的工作暂时未能收敛，大概率在对谁mask这件事上没做好，完整的Pipeline还有一个情感判别模型要参与到MLM的训练中，只能有时间再调了。

嗯，一词多义，contextual embedding v.s. word embedding，总之，感觉MLM可以用来搞很多事情，值得挖一挖。

2.《Effective Neural Solution for Multi-Criteria Word Segmentation》

在每种分词方案后添加属于该分词方案的特殊标志符。虽然是2017年的文章，但是类似思想可以用在非常多的地方，在分词任务上的提升也是非常显著的。

1.《PAWS:Paraphrase Adversaries from Word Scrambling》

主要贡献：构建了一个非常有趣的数据集。数据集的特点：两个句子，word order不一样，但是word overlap非常高；标签为语义相同/不同。

用途：bert直接作用于这样的数据集，指标非常差；通过将该类数据添加到训练数据中，可以提升模型的robustness。能够很好的处理该类数据的模型，获取non-local contextual information的能力要强。此外，使用该数据，可以很好的度量模型对于word order和syntactic structure的sensitivity。

延伸思考：该数据集是平行语料，应该有其他可能的场景。

举例如下：

(1)Flights from New York to Florida.

(2)Flights to Florida from NYC.

(3)Flights from Florida to New York.

主要技术：Language Model(按照规则构建数据后，打分过滤)+Back Translation，最终构建类型如下：

![img__](https://wx4.sinaimg.cn/mw690/aba7d18bly1g813dh8n3oj21n40f2wix.jpg)


### GNN/GCN/GAN

2.《MASKGAN: BETTER TEXT GENERATION VIA FILLING IN THE_____》

微软的MASS感觉和这篇思路很是类似。

1.《Keep It Simple: Graph Autoencoders Without Graph Convolutional Networks》

### Dialogue System

2.《PLATO：Pre-trained Dialogue Generation Model with Discrete Latent Variable》

domain code是可控文本生成的一个好的策略，具体使用方式也比较灵活。这周实现的一个工作，也有应用。

《Mask and Infill：Applying Masked Language Model to Sentiment Transfer》

1.《End-to-end LSTM-based dialog control optimized with supervised and reinforcement learning》

这个是小蜜参考的另外一篇文章。整体上两篇文章时间都算是相对较早的。

0.《A Network-based End-to-End Trainable Task-oriented Dialogue System》

陈海青在2019年的云栖大会上分享中谈到的，小蜜用的一个工作，涉及一些RL的内容。

### GAN/VAE/RL

一些关于RL（**主要是policy gradient**）用于text generation任务的文章：

5.《A Deep Reinforced Model For Abstractive Summarization》

在RL的应用上感觉没有特别亮的点，类比之前的几篇工作。


4.《A Study of Reinforcement Learning for Neural Machine Translation》

将RL用于nmt任务（单语），和之前几篇整体上类似。文章给了一个经验性的结论：

**several previous tricks such as reward shaping and baseline reward does not make signiﬁcant difference。**

3.《Sequence Level Training With Recurrent Neural Networks》,第一篇将RL用于文本生成的文章

2.《Self-critical Sequence Training for Image Captioning》[参考笔记](https://zhuanlan.zhihu.com/p/58832418)

这篇文章差不多是基于3做了一点儿小的改进。

1.《Improved Image Captioning via Policy Gradient optimization of SPIDEr》

0.《Connecting Generative Adversarial Networks and Actor-Critic Methods》, Oriol Vinyals等

从MDP的角度解释了两个方法，gan中的生成器约等于ac中的actor，gan中的判别器约等于ac中的critic。同时梳理了一些**稳定**两种方法的训练trick。**从目前的一些观察来看，要想将rl用于自己的任务，先要保证收敛，其次再谈效果。具体的任务，比如文本生成相关。**

### 技术杂谈

1.[基于深度学习的自然语言处理，边界在哪里？](https://mp.weixin.qq.com/s?__biz=MzI4MDYzNzg4Mw==&mid=2247489825&idx=5&sn=026e9257fa25bb1af2a13cab0888138f&chksm=ebb421f5dcc3a8e33463b506de142bcd4b36628977f1d095191ff68c25ef6dcab30654fdbd94&mpshare=1&scene=23&srcid&sharer_sharetime=1567267555857&sharer_shareid=0e8353dcb5f53b85da8e0afe73a0021b%23rd)

亮点：文章中在解释问题时，对应的例子真的很棒。

总结：现在这种关于在特定任务领域的DL缺陷的讨论，各家基本说辞一致，倒也没什么新鲜感。但是说归说，自己能否真正理解可能就是另外一回事了。读这类文章，印证自己的观点可能是目的之一吧。对自己相对认同的一些观点整理一下：

（1）数据量较多场景下，DL具有优势；其他情况下，传统方法胜算更大。（补充一个：简单任务下，传统方法和DL差不多。）

（2）大家心心念念的中文分词技术已经不是机器翻译领域的关键问题了，而是成为了一种建模粒度的选择。（所以，面试官问有几种分词技术的时候就基本等同问茴香豆的茴有几种写法？额，已经不吃豆子了。也劳烦面试官同学们跟进技术发展，不要难为候选人了。当然，喜欢吃豆的候选人能有选择的吃了。）

（3）句法结构多数情况下也不是问题了。（最近比较痴迷把语言学的东西融合进bert，可能任务依赖吧，没有发现显著提升。包括一些bert的理论工作证明，模型是可以在一定程度上learn到句法结构的。）

（4）“机器翻译的成功是一个比较特殊的例子，这是因为它的**源语言和目标原因的语义都是精确对应**的，所以它只要有足够的数据而并不需要其他的支撑，就能取得较好的效果。现在的自然语言处理系统大部分，还只是流于对词语符号之间的关系建模，没有对所描述的问题语义进行建模，即对客观世界建模。而人理解语言的时候，脑子里一定会形成一个客观世界的影像，并在理解影像后再用自己的语言去描述自己想说的事情。 ”（本质的讨论：现有模型学到的是啥？syntactic和semantic如何界定？我们实际需要的是semantic哦）

### Pre-trained 

9.《BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension》

FAIR的工作，同期类似工作MASS/UniLM/T5/。整体上和GPT系列类似，不过添加了BERT系组件。一图说明问题，如下：

![img_BART](https://wx3.sinaimg.cn/mw690/aba7d18bly1g8ixvzt4pjj21a20u0thi.jpg)

8.《On the Cross-lingual Transferability of Monolingual Representations》

基于Bert的嫁接技术：self-attention block和word embdding的嫁接。

7.《Learning and Evaluating General Linguistic Intelligence》

个人非常喜欢的一篇文章，讨论了一些大家似乎习以为常，但是却没有深入思考的一些问题。比如：

（1）预训练提升performance的量化分析？用downstream任务来评估总觉得不是很合适。

（2）相同的预训练任务，能否直接generalize到其他数据上？比如，SQuAD训练的模型，可以直接用于TriviaQA。

（3）fine-tune的时候，模型忘记之前学习到的knowledge有多快？

（4）curriculum是如何影响performance的，应该如何设计curriculum？

主要的研究方式：提出一个量化指标，code length，该指标与acc/f1-score等有直接关系。


6.《DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter》

主要贡献：通过蒸馏的方式，获取一个更小，inference速度更快，performance损失极少的bert。

主要方法：teacher网络就是原始较大的bert，student网络是较小的bert。损失函数=L(masked lm)+**L(cross entroy based-on soft targets)** + L(cosine distance based-on hidden representation)三个部分组成。

个人想法：

（1）student网络设计

student网络设计理论上应该比较灵活。比如选择一个三层的Transformer的Encoder端，一个一层的BiLSTM，甚至CNN系网络。对于第一种，假设用原始较大的bert中的一些层的参数做初始化，则可以认为是一个小的bert，这和文章中的做法是一致的。当然也可以选择初始化，比如对于第二种和第三种，类似工作后来在文章《Distilling Task-Specific Knowledge from BERT into Simple Neural Networks》中有看到，直觉上初始化方式和网络结构是强相关的。对于第一种，天然适合用原始bert的参数做初始化，那么对于第二种，第三种能否做参数共享值得思考。实际上，初始化方式对于蒸馏比较重要。

此外，能否直接在原始较大bert上实现类似student网络的功能也值得思考，因为这样就不需要重新去train一个新的网络，这就和bert在fine-tuning时的freeze layer/lr schedule等相关Trick相关了。

（2）损失函数设计

实际上，损失函数的设计不限于文章中所说的，本质上是如何建立teacher和student的logits之间的关系，让二者尽可能接近。比如本文的ce和（1）中提到文章的mse等。除此之外，regularization的探讨也是一个很自然的想法。

（3）蒸馏任务设计

文章的蒸馏任务是Masked LM，仍旧需要在下游任务上做fine-tuning，（1）中提到的是直接用下游任务来蒸馏。当考虑pretrained+fine-tuning的时候，蒸馏任务的设计就会比较灵活。

5.《Unified Language Model Pre-training for Natural Language Understanding and Generation》

多种预训练模式的大杂烩。如下：

![multi-pretrained-method](https://wx2.sinaimg.cn/mw690/aba7d18bly1g7oy2a23mwj212x0u0ng8.jpg)

4.《Enriching BERT with Knowledge Graph Embeddings for Document Classiﬁcation》

motivation：present a way of enriching BERT with knowledge graph embeddings（PyTorch BigGraph） and additional metadata.
任务：book classification
架构：类似于deep&wide

3.《CTRL: A CONDITIONAL TRANSFORMER LANGUAGE MODEL FOR CONTROLLABLE GENERATION》

主要贡献：

（1）带有条件的语言模型。其实，仔细看gpt的相关文章的话，已经在gpt中出现了，作者又发扬光大了。

16亿参数的语言模型，比起nvidia的80亿参数的模型较小，不过仍旧是较大的语言模型了。训练语言模型的文本如下：

Horror Text: I was a little girl when my parents got divorced. My dad had been in the military for years and he left me with my mom. She worked as an RN at a hospital so she could take care of me.\n\n When we moved to our new house it took some time before things settled down. We were still living together but there wasn’t much going on. It didn’t help that my mom would get mad if someone came over or even just walked by her house.\n\n One day while walking through the yard I noticed something out of place...

这里，Horror Text:就是条件了。

（2）penality sampling。类似于coverage机制，decoder不搞一些新的sampling策略，就感觉缺点啥；类似于不在loss上搞点事情，就觉得工作不够高大上。

（3）一个有很多参数的语言模型。16亿。

总结：散了，散了，你们玩儿吧。


2.《Pre-Training with Whole Word Masking for Chinese BERT》

BERT-wwm的工作，中文版。

motivation：wordpiece分词会导致将一个完整的词分成几个子词， 原始的bert在mask的时候会mask掉子词，但是从语义角度，更好的mask方式应该是mask一个完整的词（对比，个人持保留意见）。

solution：解决方法相对简单，就是如果发现一个词中某个子词被mask了，就全部mask掉该词对应的所有子词。

工作扩展：在给bert添加语言学信号的时候，例如pos/依存分析/srl的信号，最好考虑wordpiece分词后的对齐问题。

1.《A Robustly Optimized BERT Pretraining Approach》

facebook的工作，bert是undertrained的，并对bert重新训练的细节做了思考。

### EMNLP2019论文选读（浏览了一些自己感兴趣方向的文章）

总结：EMNLP的文章读起来八股气息要弱一些，文章类型更加丰富，个人觉得是好现象。不过收一些个人感觉明显质量有问题的工作也是有点奇怪。下述文章只是看题目觉得可看，就大致浏览了一下。

1.《Text Summarization with Pretrained Encoders》

思路：用bert去fine-tune句子级的embedding，然后对sentence做binary classification(要不要？)。实际上，直接对token做binary classification未尝不可，不过相比前者，semantic的粒度确实小了很多。一如很多工作，这次bert后不加bilstm，不加crf，加self-attention层了。额，可以一直加，但是不要搞得像CV的一些工作就行。比较有启发的是，如何构造输入进行sent embedding的学习。bert不是只可以一个[CLS]和两个[SEP]吧。

2.《Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks》

思路：当输入两个句子直接进bert得到sent embedding的问题在于，当句子比较长的时候就有点尴尬了。一个比较直接的思路是，每个bert喂一个sent，用两个bert做embedding，然后加融合层，比如cosine之类的操作等，整体上是siamese的结构。除此之外，文章也提出了triplet loss的应用。嗯，整体上的工作和人脸领域的一类工作很类似了。

3.《Neural Text Summarization: A Critical Evaluation》

个人较喜欢的一个工作，相对务虚，不过指出的问题需要思考。文章认为：**整体上神经关系抽取近期的工作基本处于停滞状态。**比如，有些数据集上，SOTA只比直接从文章中抽取前三句话好一点点。原因有三个方面：（1）数据集的问题。某些数据集只给定一篇文章+一个参考摘要，除此之外，没啥额外的信息了。真实场景下可不是这样的哦。也就是说，现在的数据集构造还是离真实场景有点远。（2）评估指标的问题。已经是生成式模型的老问题了，除了大家知道的一些弊端，还有对**事实性错误的评估**。（3）模型到底学到的是个啥？

4.《Attending to Future Tokens for Bidirectional Sequence Generation》

思路：用bert做序列生成。输入源句子+目标句子用于训练，其中目标句子随机选一些token用placeholder替换。扩展一下，关于bert用于生成，最近印象中有一些工作。第一：类似于本文的思路。第二：seq2seq中将bert作为encoder，decoder单独训练(模型可以灵活选择)。第三：decoder端用bert。第四：更加广义地应用，比如做extractive摘要的工作。

5.《Dynamic Past and Future for Neural Machine Translation》

思路：用于Capsule Network做机器翻译。

6.《Learning to Recognize Discontiguous Entities》

解决的问题：命名实体识别中，识别不连续的实体。

7.《ner and pos when nothing is capitalized》

解决的问题：在NER和POS任务中，**大小写很重要。**

8.《On NMT Search Errors and Model Errors: Cat Got Your Tongue?》

解决的问题：组合beam search和depth-first search做精确推断用于NMT任务。作者在文末明确指出：该方法不实用，但是可以帮助理解现有一些NMT的Trick可能并不能完全解决一些特定的问题。

### BERT

11.《FASPell: A Fast, Adaptable, Simple, Powerful Chinese Spell Checker Based On DAE-Decoder Paradigm》，EMNLP2019

前言：一大早老大投喂的纸。论文还没有放出来，给作者发了mail，还没回。[PR稿](https://www.jiqizhixin.com/articles/2019-10-29-8)，不过给了代码地址，还没决定要读。

主要内容：基于BERT，用MLM的方式做中文纠错。

特色：不使用烂大街技术（召回+排序），而是直接预测+后处理。直接预测部分是典型的MLM的路子，特色在后处理。整体上希望得到一组confidence和similarity的阈值，筛选出错别字。因此，简单点说：“卡个阈值做后处理。”。因此问题的关键是：confidence/similarity如何计算？前者很容易想到用prob，后者则利用了字音和字形的特征。有啥好处？比起召回+排序的路子，充分考虑了字与字的similarity。为啥是一组？这点还没想特别明白。如果是针对每个token位置计算一个，似乎make sense，但是由于序列是不等长的，也就意味着这组阈值在利用的时候也是动态的；如果是对prob排序后的行计算一组阈值，那么阈值数目固定（等于字典长度），但是似乎在similarity上不make sense。需要后续确定下细节。

问题：

（1）从文章实验数据来看，后处理对文本类纠错有性能损伤；对OCR类提升比较明显。这也是在没有看到实验结果的时候就可以想到的问题。字形的similarity想必重要性要高于字音。字音和字形理论上对文本类同音字和形近字错误纠正有帮助，但是看到了明显的损伤。

（2）文章中的实验结果和代码README.md的表格结果不一致。此外，文章中的表格数据方差也太大了。

（3）召回+排序 还是 直接预测+后处理，是围绕MLM的在技术路线上的选择。前者更加的可控，不过也不太灵活；后者很灵活，但是可控性较差。具体使用哪种，还是要具体问题具体分析。其实，本质上具有一致性，就是filter的操作放在模型前做，还是放在模型后做的问题。简单点说：“前处理还是后处理”。围绕这个选择，最近真的是有好几个工作，组里也是有不同的尝试。


10.《Exploiting BERT for End-to-End Aspect-based Sentiment Analysis》

干货不多。有一个问题值得思考：Does BERT-based model tend to **overfit the small training set**?

文章认为：F1-score on the development set are quite stable and do not decrease much as the training proceeds, which shows that the BERT-based model is exceptionally robust to overfitting.

9.《Semantics-aware BERT for Language Understanding》

个人理解这篇工作比较工业风，核心观点：把语言学知识融入到模型中。类似工作非常多并且适合灌水，比如融入NMT/Dialogue等。这里语言学：SRL;模型：Bert。技术上的问题：SRL结果较多需要融合，Bert的SubWord分词和SRL输出结果的对齐，模型结构的对齐，和原始Bert Word Emb的融合等。其实挺没劲的，不过挂了非常多的作者。词法，句法，语义都可以灌，逃。

8.《Universal Text Representation from BERT: An Empirical Study》

讨论了各种Bert抽Sentence Embedding的方法，比较系统。

7.《ALBERT：A Lite BERT For Self-Supervised Learning Of Language Representations》，iclr2020投稿

结论：参数更少，效果更好的bert变种。

主要工作：

（1）降低embedding的dim，O(vxh)->O(vxe+exh) where h>>e（隐含条件v>>e），是一个较简单且松弛的不等式问题；

（2）跨层参数共享。类RNN，将layer视为building block。作者也尝试了共享ffn和共享attention

（3）sop任务。原来的nsp是鸡肋，因为有信号泄漏，实际上模型可能没有学到多少coherent信息，而是泄漏的topic信息；这里重新设计了一个任务：
给定一篇doc中的两个句子a和b，其中a和b按照doc中的顺序出现，则a+b=1；b+a=0。这样可以push模型学到coherent信息而不是topic信息。

评价：总体上看没有骚操作，不过工作做的很是干净，效果也很不错。

6.5和4的相关paper，《Enriching BERT with Knowledge Graph Embeddings for Document Classiﬁcation》，德国

5.《ERNIE: Enhanced Representation through Knowledge Integration》，百度

4.《ERNIE: Enhanced Language Representation with Informative Entities》，清华

3.《Revealing the Dark Secrets of BERT》

model pruning 

