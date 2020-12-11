252.《Why Are Deep Learning Models Not Consistently Winning Recommender Systems Competitions Yet?》，RecSys2020

非常棒的文章，多年以前自己就很好奇了。

251.《Enriching Pre-trained Language Model with Entity Information for Relation Classiﬁcation》

![img_251](https://ftp.bmp.ovh/imgs/2020/11/fd20de4ee9f75cfc.png)

250.《Diverse, Controllable, and Keyphrase-Aware:
A Corpus and Method for News Multi-Headline Generation》

新闻标题生成

249.《CharBERT: Character-aware Pre-trained Language Model》

248.《A Frustratingly Easy Approach for Joint Entity and Relation Extraction》

![img248](https://ftp.bmp.ovh/imgs/2020/11/34e4c031463fcc48.jpg)


247.《Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba》

基于Graph的结构，通过Graph Learning的方式获取商品Embedding。这里的Graph是基于历史订单共现数据搭建的（我们自己的搭建方式是基于商品共有属性），整体的思路是类似于Word2Vec。

![img247](https://wx4.sinaimg.cn/mw690/aba7d18bly1gk9s6fxsdhj211a0d679c.jpg)

246.《Iterative Strategy for Named Entity Recognition with Imperfect Annotations》，NLPCC2020的AutoIE赛道的题目

245.《Pretrained Transformers for Text Ranking: BERT and Beyond》，综述文章

244.《Multimodal Attribute Extraction》

现在NLP的多数属性抽取任务是通过直接分析文本的基础上做的，这篇文章在做属性抽取的时候，融合图片的信息。（对多模态不是很了解，很好奇图片的信息真的能够帮助提效吗？）

243.《中文电子病历命名实体和实体关系语料库构建》，2016，软件学报

非常棒的文章。可能不像现在多数数据集构建的文章，会跑一部分的baseline，但是这篇文章的特点更加明显，有用。

（1）总结了诊疗过程中的研究对象

（2）阐述了数据集构建的具体流程和标准等

（3）给出了建模的基本方法

（4）总结了数据构建过程中的经验和教训

值得多看，对其他领域数据集构建也有启发。

242.《MetaPAD: Meta Paern Discovery from Massive Text Corpora》

从大规模语料中挖掘类似于下图左侧的meta pattern。

![img_242](https://wx2.sinaimg.cn/mw690/aba7d18bly1gjmynkys55j20np0gadka.jpg)

241.《Simplify the Usage of Lexicon in Chinese NER》

KeyPoints：如何在做NER的时候，利用lexicon的信息？（Xuanjing Huang老师组的一直在做的Topic）

工作延续：是对Lattice-LSTM的进一步优化，相关工作还有一些，主要解决：在保证metric不变的前提下，提升inference speed。

做法：利用分词（BMEO）+词频（作为权重）

Motivation:分词信息能够完整表征用词典去match句子时得到的匹配信息；Attention是一种dynamic weighting技术，词频是一种静态weighting技术（预计算），类似想法在ZEN的工作中也有体现。

直观感受：一种NER任务的Trick实现。围绕Lattice-LSTM的工作，最近也有一个工作是基于Transformer来做的。

240.《Unsupervised Text Style Transfer with Padded Masked Language Models》

239.《Tuning Word2vec for Large Scale Recommendation Systems》，Twitter

Word2vec用于推荐系统（基于用户行为序列的表征）

238.《What if we had no Wikipedia? Domain-independent Term Extraction from a Large News Corpus》

domain内的自动术语抽取。文章中提到一个有趣的场景：wikipedia的词条哪些是需要编辑的？（人类知识是动态可进化的）

237.《Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks》

讨论BERT系用于domain任务的时候，是否需要进一步pre-training，具体怎么做？在早期，UniLM的工作中讨论过类似做法，去年的一些工作中也有尝试，但是整体结论是：提升是不可预期的，因此个人在后期的一些任务中，多采用直接finetune，没有做进一步的pre-training。

236.《The Renaissance of Templates for Data2Text: A Template Based Data2Text System Powered by Text Stitch Model》，EMNLP2020

这篇想读，但是文章还没有share。

235.《FLAT: Chinese NER Using Flat-Lattice Transformer》,ACL2020

中文NER目前的SOTA，利用分词信息增强BERT，个人比较喜欢的工作，比较干净。黄老师组应该一直在做lexicion-based的NER工作，之前做过PU Learning用于NER的相关工作。[相关参考](https://mp.weixin.qq.com/s?__biz=MzIwNzc2NTk0NQ==&mid=2247497102&idx=1&sn=cedddfa134b0a2e0ca30b3f033560eb2&chksm=970fdd58a078544e75a84939682ca36a3dc412c93853f5377d3a17dcc3c5c399567994a34ea1&scene=178#rd)，位置信息的利用采用position-encoding，但是和《Attention is all you need》中的工作不同。
![img_235](https://wx1.sinaimg.cn/mw690/aba7d18bly1gikrga7al8j20hq0jdgoj.jpg)


234.《Adapting BERT to E-commerce with Adaptive Hybrid Masking and Neighbor Product Reconstruction》

电商领域的BERT。

核心思想：融入知识到BERT中去，不同领域有不同的知识，电商就是一个垂直领域。

知识：常识，图谱，词典等。

结论：在理解任务上提升明显。

门槛：知识沉淀。

233.《Dynamic Updating of the Knowledge Base for a Large-Scale Question Answering System》

主要内容：自动化问答库构建

作者：晓多科技（智能客服赛道的大玩家），江岭

亮点：有真实线上数据反馈

232.《Match 2 : A Matching over Matching Model for Similar Question Identification》,SIGIR2020

核心思想：在计算QQ相似性的时候，用A做bridge。

想法：该思想可以体现在很多地方，比如triplet loss（anchor），比如rank loss，比如特征交叉（融合）等。

三种范式如下(文章主要讨论的是第三种)：

![img_232](https://wx3.sinaimg.cn/mw690/aba7d18bly1gigvfk7d72j21da0cttbp.jpg)

231.《DC-BERT: Decoupling Question and Document for Efficient Contextual Encoding》，SIGIR2020

解耦quesiton和passage的编码，加速。（SIGIR的文章，总体感觉质量一般）

230.《MarkedBERT: Integrating Traditional IR Cues in Pre-trained Language Models for Passage Retrieva》，SIGIR2020

找到question和passage的共有实体，加marker。

229.《It’s better to say “I can’t answer” than answering incorrectly: Towards Safety critical NLP systems》

概率校准。

KDD2020的一个tutorial：https://github.com/nplan-io/kdd2020-calibration

228.《Knowledge Eﬃcient Deep Learning for Natural Language Processing》

还没细致读，Ph.D论文，解决的问题：如何引入外部知识，减少标注数据量，从而提升模型效果。个人非常感兴趣的方向。

提出一个概念：知识高效学习，具体包括技术：

（1）多任务学习

（2）迁移学习

（3）弱监督学习

（4）无监督学习

227.《Rethinking the objectives of extractive question answering》

核心思想：由对start和end的单独建模，改为联合建模。

相关Trick：

（1）Distant Semi-Supervision

（2）Top-K surface form ﬁltering

（3）length ﬁltering

226.《AMBERT: A PRE-TRAINED LANGUAGE MODEL WITH MULTI-GRAINED TOKENIZATION》，李航

思想：整合多个句子的分词粒度，能够在中文任务（CLUE）上带来显著提升。有些词词频很高，导致Attention在一定程度上会关注词频比较高的组合，但是在给定context的时候，也许并不希望这样。

文章尝试了三种方式：

（1）给定一个句子，直接合并两种分词方式下的结果，作为模型输入。（共享Encoder端）

（2）两种分词方式分别用两个不共享的Encoder编码

（3）两种分词方式共享Encoder，带来了最大的提升。损失函数的设计除了传统的Mask Loss，还要考虑两种分词方式得到表征的交互（两项，其中一项是关于对[CLS]的正则）

想法：

（1）观点不算新。

（2）文本匹配的经典思路在BERT上的一种体现。第一种是cross encoder的经典范式；第二种是siamese的思想体现；第三种是shared encoder；

（3）中文理解任务可以考虑用AMBERT替换Roberta了

225.《Progress in Neural NLP: Modeling, Learning, and Reasoning》

最近MSRA的周明，段楠写的一篇NLP的综述文章。围绕modeling，learning和reasoning三部分讨论。

第一部分主要讨论NLP的主要模型结构。如何设计一个好的模型结构呢？（RNN/CNN/Self-Attention）

第二部分主要讨论学习策略。比如transfer learning/multi-task learning等

第三部分主要讨论啥是NLP中的推理（**个人觉得这部分比较值得看**）。涉及图谱/常识，经典方法（整数线性规划，马尔科夫逻辑网，基于记忆的网络等），同时结合具体case，讨论了方法的原理。

224.《Cognitive Representation Learning of Self-Media Online
Article Quality》

微信自媒体文章质量评估。在早期，有不少工作是做网页质量评估，透出应用是给搜索提供一个很强的feature。

223.《Reinforcement Learning to Rank in E-Commerce Search Engine: Formalization, Analysis, and Application》

RL在电商搜索场景中的应用，显著提升了GMV。（南大俞扬老师参与，大概在2017年阿里出了一份报告：强化学习在电商领域的落地应用，俞老师也在某些场合聊过RL可以落地，印象中包括和滴滴的合作。）

222.《Long-Tail Learning via Logit Adjustment》

基于损失函数变种解决长尾学习问题。

221.《Embedding-based Retrieval in Facebook Search》

siamese network+triplet loss+negative/positive sampling(显著提升点)+feature engineering+cascade methods+ANN

[网友笔记](https://mp.weixin.qq.com/s?__biz=MzIwNzc2NTk0NQ==&mid=2247495579&idx=1&sn=1308770beb440e76e2004d94b22347ce&chksm=970fc74da0784e5b74a58436416d6cf71ad170e284a2c678b7b4ca373262ce2481a395246f3c&mpshare=1&scene=23&srcid=0804Spa7PkMvRAYpIFN3EepC&sharer_sharetime=1596551325870&sharer_shareid=0e8353dcb5f53b85da8e0afe73a0021b%23rd)

220.《Asking Clarifying Questions in Open-Domain Information-Seeking Conversations》，SIGIR2019

问题澄清，从IR的角度来解，构建了一个基于IR的数据集。其中，比较了传统的L2R的方式，同时给出了相关与IR相关的评估指标：MAP/MRR/NDCG。

关于问题澄清的业务思考：

（1）当人询问机器的时候，如果得不到想要的答案，会换种说法问问题，是人适应机器的过程；

（2）本质上对话是一个交互的过程，在人机对话的过程中，机器也要猜想人的意图，这点正是需要深入思考的。

219.《Generating Natural Answers by Incorporating Copying and Retrieving Mechanisms in Sequence-to-Sequence Learning》

现在看到这种工作，第一个想的是：如何兜底？出现Badcase如何修？不考虑这些问题，技术上确实是fancy的，但是实际应用的时候，如果没有兜底方案，那就很糟糕了。另外，不是所有问题都有很好的兜底方案。

![img_219](https://github.com/zhpmatrix/zhpmatrix.github.io/blob/master/images/COREQA.png?raw=true)

218.《Predictive Multiplicity in Classiﬁcation》，ICML2020

为模型评估增加一个新的评估维度，在去年讨论较多的是模型的碳排放量，属于指标体系完善的一类工作。

217.《SoftSort: A Continuous Relaxation for the argsort Operator》，ICML2020

argsort的可微分实现，类似工作之前也有，但是这里是更加简单，更加高效的实现；在很久之前，有一些工作是针对metric做可微分优化的，但是实际上，似乎没有成为一种成熟的技术被推广开来。

216.《Deep Streaming Label Learning》，ICML2020

连续学习相关工作。

215.《AILA: A Question Answering System in the Legal Domain》，IJCAI2020

一个法律问答系统。

214.《A Unified Model for Financial Event Classification, Detection and Summarization》，IJCAI2020

multi-task的方式，一个模型（bert）解决多个问题，问题域：金融。

213.《Response Generation by Context-aware Prototype Editing》

响应生成一类经典的方法：结合检索+生成的方式。

212.《Language-agnostic BERT Sentence Embedding》

基于平行翻译语料，使用additive margin softmax训练的表征模型。可以用于从大规模语料中挖掘平行语料对，用于翻译模型训练。

211.ACL2020的最佳论文，《Beyond Accuracy: Behavioral Testing of NLP Models with CheckList》

有谁像我一样，第一次看到的时候，直接略过了。[允悲]在去年的一段时间，遇到模型上线难的问题，体现在多个方面：提升不明显的时候，冒烟测试可能根本过不了，甚至体验会下降；评估标准是多维度的，存在指标冲突的情况；Trade Off的事情不是容易的。产品经常会拿一些badcase报给算法，立刻fix掉还是周期性fix，是个问题。当时考虑的问题是：如何全面评估模型的问题？显然，benchmark是统计上的指标，产品同学可不这样看。因此，在之前的微博中也提到过从软件工程中借鉴一些测试的方法论。但是针对模型测试，要确定的是测试的维度。这篇文章就给出了一些比较general的测试维度，算是系统的模型测试方法论，但是文章没有回答的是：如何fix？关于fix的方式且看各种文章吧，有针对general做优化的，有针对特殊case做优化的。

在文章中，值得注意的是提到了黑盒测试所不能覆盖的方面。

这个工作感觉也可以产品化。基本思路是：数据诊断+模型诊断；诊断类型包括NLP，CV和语音（灌水看这里，CV和语音想必还没有类似工作）；可以做多平台多模型测试（一个用户痛点是：好多身边不做技术的朋友来问我，哪家的某个API比较好，模型界的王自健），站在产品的角度考察，完成的测试要考虑的因素就更多了，这篇文章也只考虑了一个比较大的维度。

最近琢磨的几个工作，算是和这类工作有相同感觉：

（1）吕正东：符号+神经的工作（不是那种模型+规则的路子）

（2）正则+神经的工作（FSL领域有一篇工作）

总体上，这篇工作还是比较实用的，可能大家都觉得是问题，但是没有系统的去做过梳理。

210.《SpanMlt-A Span-based Multi-Task Learning Framework for Pair-wise Aspect and Opinion Terms Extraction》，阿里ACL2020的工作

在去年的”之江杯“电商评论观点挖掘比赛中，定边界+组合pair做01关系分类已经有较多工作了，自己看到过pipeline的方式，这篇感觉使用了multi-task的方式。总体上，感觉没啥亮点，不过可以作为一个做切面情感分析的Trick来使用。

![img_210](https://wx1.sinaimg.cn/mw690/aba7d18bly1ggjdexdrofj20wo0e0teb.jpg)

209.《Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering》

基于生成模型做开放式领域问答（本质上知识是存储在模型参数中的），分为两步：

第一步：给定问题，检索出相关文本片段

第二步：给定问题和相关文本片段，直接生成答案

208.《Simple and Effective Text Matching with Richer Alignment Features》，ACL2019

简单快速的文本匹配模型，实测快，效果稳。亮点是：对文本匹配的关键组件进行了思考。对比bert，两个模型会关注不同的aspect，理论上配合使用较好。

207.FSL的两篇文章：

《Improving Few-shot Text Classiﬁcation via Pretrained Language Representations》

《When Low Resource NLP Meets Unsupervised Language Model: Meta-pretraining Then Meta-learning for Few-shot Text Classiﬁcation》

两篇文章同一拨人，感觉是一个工作写了两次。主要思想和205谈到的思路不同，和传统认识中的meta-learning比较类似，整体上看方案显得比较simple，但是提升比较明显。


206.《AutoML: A Survey of the State-of-the-Art》

AutoML的特色在NAS，同时存在一些自学习平台，比如达摩院的，和Google的，注意二者之间的区别。

![img_206](https://wx3.sinaimg.cn/mw690/aba7d18bly1gg7v92iedaj20x30eoaci.jpg)

205.FSL的两篇文章：

A纸：《Induction Networks for Few-Shot Text Classiﬁcation》，B纸：《Dynamic Memory Induction Networks for Few-Shot Text Classiﬁcation》，同一拨人的工作。上图是A的结果，中图是B的结果，下图解释了为啥baseline会高两个百分点，自己和自己的工作比，这样就很好，又不费事。证明了BERT还是强，不搞Trick，稳稳两个百分点的提升，不过这样就显得没啥insight了，不过工业界喜欢。但是另外的问题来了，A和B都有两个数据集，一个是公司内部的不能share无可厚非，但是为啥A用了一个，B用了另外一个呢？效果不好还是不能使用？（没说明，我还没想。。）。另外一个问题，搞FSL的都是C-kay-K-shot，实验都是C和K定了再给结论的吗？（好像对于挑出来的C和K，FSL的工作都不刻意说明。。超参吧？！），不过同一个topic，看同一拨人做，前后对比起来看，还是有启发的。

![205_1](https://wx2.sinaimg.cn/mw690/aba7d18bly1gg65az6x2ij20q507qac5.jpg)

![205_2](https://wx3.sinaimg.cn/mw690/aba7d18bly1gg65b1kqc3j20fw09ddha.jpg)

![205_3](https://wx2.sinaimg.cn/mw690/aba7d18bly1gg65b5b904j20fx08o770.jpg)


204.《Reinforcement Learning for User Intent Prediction in Customer Service Bots》，SIGIR2019

这篇文章是蚂蚁智能客服中“猜你想问”的一种实现方式，该问题是一个经典的Top-N推荐问题，这里转化为N步序列决策过程，本质上是学习多模型融合时的weight该如何分配的问题（这里主要是一些ranking模型），这种范式在互联网的很多场景下都是可以的，不过这里是用在了智能客服中的意图识别相关任务中。

其他应用包括：

（1）基于强化学习的交互式文本推荐

（2）基于强化学习的交互式澄清

（3）基于强化学习的客户路由

203.两篇关于NLP模型robustness的文章：

《Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and Entailment》

《Combating Adversarial Misspellings with Robust Word Recognition》

robustness一般讨论攻和防两端，此外，与数据增强也有一定的联系，不过最终的目的总是：更好的泛化能力。

202.《PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization》，ICML2020

重点考虑为了更好地满足生成式摘要设计的自监督任务。

![img202](https://wx4.sinaimg.cn/mw690/aba7d18bly1gg4xseqqnvj20gk0d6q51.jpg)

201.《Recipes for building an open-domain chatbot》，ParlAI的工作

讨论对比了开放域对话的检索式，生成式，以及检索式+生成式三种方案。

200.《Dialog Policy Learning for Joint Clariﬁcation and Active Learning Queries》

Joint建模方式的几个观察：

（1）两个任务建模。比如这篇，比如联合意图识别和实体抽取

（2）多个任务建模。

（3）非常多任务建模。比如用seq2seq，mrc的思路统一建模。

本质上，希望一个模型能够做更多的事情，具体操作上有所不同。

199.《Towards Uniﬁed Dialogue System Evaluation: A Comprehensive Analysis of Current Evaluation Protocols》

闲聊对话系统的评估维度总结。

198.《Data Augmentation for Training Dialog Models Robust to Speech Recognition Errors》

ASR的错误一定要纠正之后，才能用于下游模型吗？关键看如何理解这些错误。

197.《Cross-domain Aspect Category Transfer and Detection via Traceable Heterogeneous Graph Representation Learning》，CIKM2019

核心思想：利用电商场景下的用户行为，构造一个Graph（包含商品，用户，卖家等信息和行为），学习Graph的表示，用于评论切面检测任务。

个人想法：一个Graph的好的表示可以将多个研究对象纳入一个统一的表示空间（Pattern的量化），理论上可以用于很多下游任务。这篇文章选择了一个切面检测任务，不过看整体评测指标不算高，个人觉得有学术价值。除此之外，该工作与Few-Shot Learning也有关系，一定程度上有助于cold start问题的解决。

196.《POSITIONAL MASKING FOR LANGUAGE MODELS》

核心思想:不仅mask token，position也可以mask。

195.最近两篇关于将BERT信息压缩到一个卷积网络的工作：

第一：将BERT的压缩到TextCNN，文章忘了。

第二：《Accelerating Natural Language Understanding in Task-Oriented Dialog》（ACL 2020 Workshop on NLP for Conversational AI），将DistillBERT压缩到一个CNN网络，用于Dialogue场景。

194.《Maximizing Subset Accuracy with Recurrent Neural Networks in Multi-label Classification》

和SGM的思想类似，但是SGM没有引用这篇文章。用seq2seq做multi-label的问题。

193.《PoWER-BERT: Accelerating BERT inference for Classification Tasks》

**The method works by eliminating
word-vectors (intermediate vector outputs) from the encoder pipeline. We design a strategy for
measuring the significance of the word-vectors based on the self-attention mechanism of the
encoders which helps us identify the word-vectors to be eliminated**

补充：

《DeFormer: Decomposing Pre-trained Transformers for Faster Question Answering》，ACL2020，加速问答模型

192.《Dense Passage Retrieval for Open-Domain Question Answering》，ACL2020

想起来ACL2017的文章《Reading Wikipedia to Answer Open-Domain Questions》，这两篇陈丹琦都在作者群中。但是上篇我
