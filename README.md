229.《It’s better to say “I can’t answer” than answering incorrectly: Towards Safety critical NLP systems》

概率校准。

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

想起来ACL2017的文章《Reading Wikipedia to Answer Open-Domain Questions》，这两篇陈丹琦都在作者群中。但是上篇我理解将retrieval阶段由这篇的bm25/tf-idf换成了基于向量检索的方式（个人之前是基于SentenceBert做的），可以使用FAISS等向量检索服务。

所以：为啥中了ACL2020？

191.《Syntactic Search by Example》

个人非常喜欢的工作。[demo](https://spike.wikipedia.apps.allenai.org/search/wikipedia/#query=Ok1pY3Jvc29mdCB3YXMgJGZvdW5kZWQgJGJ5IDpQYXVs&queryType=S&autorun=true)

190.WWW2020的Best Paper，《Open Intent Extraction from Natural Language Interactions》

（从很久之前的某条微博搬来的）刚好match最近做的一些工作。第一个，OpenIE相关，最近参与做的类似Magi一样的工作,本质上是一个schema设计的问题。之前做意图分类，是Closed域内的问题，这篇文章通过针对意图的schema设计，用序列标注的思路抽取意图，一样的思路。不过存在的问题是，有些意图不会在对话上下文中存在明显的字符串，这和supervised oie与supervised ie之间的问题一样；另一方面，任务型Chatbot的落地方案中，特色在于表的设计，现在还是人工设计，表通常包括三个维度：领域，意图和槽。这篇文章其实一定程度上将意图的自动化推进了一步，领域和槽的自动化设计尚未涉及。最后，文章在序列标注上还是经典的BiLSTM+CRF，不过还有一些基于CRF的比较实用的Trick。总之，大致浏览下文章，甚和我意。不一定是多牛的工作，但是让人很舒服的工作。

补充（新补充）：《Automatic Discovery of Novel Intents &amp; Domains from Text Utterances》

189.《Exploring Cross-sentence Contexts for Named Entity Recognition with BERT》

做法：句子间平滑。（方法简单有效），做法如下：

![img_189](https://github.com/zhpmatrix/zhpmatrix.github.io/blob/master/images/cmv.png?raw=true)

188.《Evaluating Natural Language Generation with BLEURT》

NLG生成新指标：BLEURT，感觉还是稍重的方案。

187.《Language Models are Few-Shot Learners》

GPT-3的原始论文。做了啥事？**更大的GPT，以至于可以不用fine-tune了**。（似乎应该可以想到）关于是否是因为训练数据过大，导致模型记住了数据而已，文章中也有讨论。

186.《CERT: Contrastive Self-supervised Learning for Language Understanding》

提出一种新的contrastive的SSL范式。**针对原始样本，生成该样本的增强样本，两两判断增强样本是否来自一个原始样本。**

185.《GECToR – Grammatical Error Correction: Tag, Not Rewrite》

标注方法类似于LaserTagger，基于序列标注的思路求解。通过迭代序列标注的方式做纠错。仍旧是基于合成数据，并且指标较低。

184.《Pretraining with Contrastive Sentence Objectives Improves Discourse Performance of Language Models》，**Dan Jurafsky**

这篇文章提到了一种预训练模型的技术，该技术可以提升模型获取篇章级表示的能力。一个简单的思想是：设计更大的上下文学习任务（句子级别）。

183.《Stronger Baselines for Grammatical Error Correction Using Pretrained Encoder–Decoder Model》

基于合成数据做的。

182.《Efﬁcient strategies for hierarchical text classiﬁcation: External knowledge and auxiliary tasks》，ACL2020

用seq2seq来modeling层次文本分类。

181.《Masked Language Model Scoring》，ACL2020

一种基于MLM的计算PPL的方式，比较重的方案。

![img_181](https://github.com/zhpmatrix/zhpmatrix.github.io/blob/master/images/mlm_ppl.png?raw=true)

180.《Dialogue-Based Relation Extraction》，ACL2020

基于对话的关系抽取数据集，对话场景下的**信息抽取**在不同的任务中有不同的表示方式。

179.《Spelling Error Correction with Soft-Masked BERT》，ACL2020

检错模型（轻量级的GRU）+纠错模型（重量级的BERT），整体上是序列标注的思路

[Soft-Masked BERT:文本纠错和BERT的最新结合](https://mp.weixin.qq.com/s?__biz=MzI4MDYzNzg4Mw==&mid=2247498877&idx=4&sn=8720f5d8875484b5acc36d65d2859b0f&chksm=ebb7c4a9dcc04dbfcca21a3053e63578ae1b1a71649189ef0b0f5cce689906b5e219cf5f11e9&mpshare=1&scene=23&srcid&sharer_sharetime=1591150207165&sharer_shareid=0e8353dcb5f53b85da8e0afe73a0021b%23rd)

178.《Contextual Embeddings: When Are They Worth It?》,ACL2020

讨论一个问题：什么时候使用contextual embedding，什么时候使用static embedding？

static embedding：语言的变化性不多，数据标注丰富；

contextual embedding：language containing complex structure, ambiguous word usage, words unseen in training;

177.《Table Search Using a Deep Contextualized Language Model》

任务上有趣：表格搜索；但是，收获很小。

176.《Conversational Word Embedding for Retrieval-Based Dialog System》,ACL2020

训练数据：<post,reply>对

模型：传统词向量训练模型+微修

用法：单独使用和传统的Embedding一同使用

175.《Beyond Accuracy: Behavioral Testing of NLP Models with CheckList》，ACL2020

一种NLP中的行为驱动测试实现。

174.《Iterative Memory-Based Joint Open Information Extraction》，ACL2020

做开放信息抽取的工作，也就是开放SPO抽取相关。主要包含两个工作：

（1）无监督的方式搞数据；（一个score&filter方案）

（2）生成的方式生成多个spo；decoder端每次只去生成一个spo，然后将生成的spo和原始输入做融合，生成第二个spo；

173.《DIET: Lightweight Language Understanding for Dialogue Systems》

rasa内置的一个intent classification和entity extraction结合做的模型。想法上比较有特色的是：

（1）结合masked方式做训练

（2）对label进行embedding，similarity作为loss的输入。而非传统的不对label做embedding，直接算ce loss；

![173_img](https://github.com/zhpmatrix/zhpmatrix.github.io/blob/master/images/diet.png?raw=true)

172.《End-to-end LSTM-based dialog control optimized with supervised and reinforcement learning》

比较早的文章了，第一篇用end2end的方式做task-oriented的bot。supervised learning可以用较少的数据给reinforcement learning提供一个好的initial state。一般而言，玩法是建立在一个多分类任务上，对history的利用是关键。

171.《Enabling Language Models to Fill in the Blanks》

提出一种预训练LM的输入/输出构建方式。输入是包含Blank的句子，输出是输入和Blank对应Token的并（用Answer符号显式拼接）。对比T5，采用seq2seq，输出端不包含输入。一种想法是：输出包含输入，在作用上可以类比seq2seq中的encoder的作用，好处是不需要一个单独的encoder。因此也就能够讲得通T5在输出端不需要包含输入。该工作用于故事生成，用Blank替换一段故事描述，采用预训练LM生成该描述，类比于改写的工作。

至此，对于包含Blank的输入，输出如何构建才能得到一个好的预训练LM，看到的有以下方式：

（1）从左到右生成原始句子

（2）从右到左生成原始句子

（3）输入和原始句子的并

（4）输入和Blank对应Token的并（用Answer符号显式拼接）

（5）Blank对应Token的并（用Answer符号显式拼接，类比T5）

170.《A Simple Framework for Opinion Summarization》

169.《Fine-grained Fact Veriﬁcation with Kernel Graph Attention Network》，ACL2020

之前做过一个事实性校验的工作。典型的场景是这样的：给定一句话包含对一个人物的描述，如”zhpmatrix在杭州工作，做NLP方向的工作，balabala...“，也就是说这句话能够精准定位一个人物：zhpmatrix，但是这句话中可能某个地方错了（事实性错误），现在要检查并修正这个事实性错误。

一般的思路是：需要一个参照上下文，这个上下文的存在方式可以是知识图谱，可以是非结构化的对该人物的描述等。如果是知识图谱，则存在实体消歧的问题，会引入另外一个模型；这篇文章采用的思路是后者。

modeling：建模为一个多分类问题。（KGA是这篇工作的内容，个人不是特别感兴趣。）

![img_169](https://github.com/zhpmatrix/zhpmatrix.github.io/blob/master/images/fact%20verification.png?raw=true)

168.《On the Robustness of Language Encoders against Grammatical Errors》，ACL2020

讨论BERT系用于中文纠错的robustness问题（这个问题在基于BERT的相关工作中基本都可以看到，不过不同任务对robustness的敏感度不同，比如文本分类一般认为是对于鲁棒性不敏感的任务），解决的思路：adversarial learning（其实也是一个常见的思路了）。具体方法：构建一些样本（如何构建是关键）和原始训练样本一块训练。

167.《DCR-Net: A Deep Co-Interactive Relation Network for Joint Dialog Act Recognition and Sentiment Classiﬁcation》

joint learning with dialog act recognition and sentiment classification

166.《Mapping Natural Language Instructions to Mobile UI Action Sequences》ACL2020

解决的任务：

![img_166_0](https://github.com/zhpmatrix/zhpmatrix.github.io/blob/master/images/acl2020.png?raw=true)

解决的方法：（相似的思想：如何将序列标注任务转化为一个seq2seq呢？在今天组里的分享中同样提到这个观点）

![img_166_1](https://github.com/zhpmatrix/zhpmatrix.github.io/blob/master/images/acl2020_.png?raw=true)


165.知识驱动对话的两个应用工作：

《A Knowledge-Grounded Neural Conversation Model》

![img165_0](https://github.com/zhpmatrix/zhpmatrix.github.io/blob/master/images/knowledge_driven_baseline_0.png?raw=true)

《Learning to select knowledge for response generation in dialog systems》

![img165_1](https://github.com/zhpmatrix/zhpmatrix.github.io/blob/master/images/knowledge_driven_baseline_1.png?raw=true)

164.《A Survey on Dialog Management: Recent Advances and Challenges》

小蜜北京团队最新关于对话管理的综述文章。讨论了**scalability，data sparsity, training efficiency**的三个问题。总结：工业界还是写if...else...，研究上可以做一些RL的探索，最近蚂蚁做对话的人来公司交流的时候，说他们就在做类似工作。

163.《StructBERT:Incorporating Language Structures into Pre-training for Deep Language Understanding》,ICLR2020

达摩院的关于bert的工作，作为基础模型在推广。

162.《Enriched Pre-trained Transformers for Joint Slot Filling and Intent Detection》

slot filling and intent classification同时做，基本是在达摩院之前的工作基础上加了一些东西。

![img_162](https://github.com/zhpmatrix/zhpmatrix.github.io/blob/master/images/Enriched%20Pre-trained%20Transformers%20for%20Joint%20Slot%20Filling%20and%20Intent%20Detection.png?raw=true)


161.《Look at the First Sentence: Position Bias in Question Answering》

非常有意思的文章，讨论了QA中的positional bias问题。

160.《Code-Switching for Enhancing NMT with Pre-Specified Translation》，NAACL2019

很实用的工作。利用用户词典和电商术语库提升翻译质量。简单来说，之前的方式是用一个特殊符号占位，翻译对应的词；特殊符号和原始上下文在一定程度上会破坏原始的语义信息。这里采用的是另外一种方式(假设中英翻译，其实是另外一种占位，不过semantic上似乎更加合理一些)：

原文：病越来越厉害

中间原文：sick越来越厉害

翻译结果：sick is worse

总结：**机器翻译中的干预机制**是一个小方向，非常具有实用价值，相关工作应该有不少。

159.《How Does NLP Benefit Legal System: A Summary of Legal Artificial
Intelligence》ACL2020

比赛，相关文章，还有这篇，法律领域的PR稿？

158.《BLEU Neighbors: A Reference-less Approach to Automatic Evaluation》

梳理了相关评价指标的一些工作。

157.《LightPAFF: A Two-Stage Distillation Framework for Pre-training and finetuning》

这篇文章的技术思路风格和之前读过的一篇关于BT的文章很是类似。

156.《Coach: A Coarse-to-Fine Approach for Cross-domain Slot Filling》,ACL2020

文中的template regularization比较有意思。整体上的思路是coarse-to-fine，是实体识别中的标准范式（比如通常第一步是边界预测，第二步是类型预测）；

155.《Learning to Rank with BERT in TF-Rankings》

最近做问答，一个比较general的框架最后是一个ranking模型，这篇文章没有单独将ranking模型剥离出来，而是和BERT一块modeling，思路上有启发性。

![img155](https://github.com/zhpmatrix/zhpmatrix.github.io/blob/master/images/bert_with_tf_ranking.png?raw=true)

154. 融合词典的信息用于NER的三篇工作，黄萱菁老师组感觉对这个topic很感兴趣。去年ACL2019有篇PU Learning搞的。

《Chinese NER Using Flat-Lattice Transformer》ACL2020

把Lattice LSTM中对词的利用拍平。

《Simplify the Usage of Lexicon in Chinese NER》，ACL2020

工作在思路上类似ZEN。

相关工作：

《Enhancing Pre-trained Chinese Character Representation with Word-aligned Attention》

153.《Learning to Classify Intents and Slot Labels Given a Handful of Examples》

解决Chatbot中的low resource setting下的问题。核心思路：MAML和原型网络+pre-trained model用于意图分类和槽填充。

152.《Distilling Knowledge for Fast Retrieval-based Chat-bots》SIGIR2020

关于知识蒸馏的，把BERT的知识蒸馏到BiLSTM中。其中有两个小Trick，一个是带上原始的标签；另一个是算MSE；都是Loss端的优化。最简单的形式是只要soft标签+ce损失。（自己用12层BERT蒸6层，蒸Albert。）

151.《SKEP: Sentiment Knowledge Enhanced Pre-training for Sentiment Analysis》，ACL2020，Baidu

核心思想：对aspect和sent word进行mask。

套路：Baidu的同学说，我们能预训练，想要啥，就对啥进行mask，你看，刷了情感的sota吧？

我：鼓掌。。

![151_img](https://mmbiz.qpic.cn/mmbiz_jpg/uYIC4meJTZ0vad7YcUGC4WTAgya8zHRvsPk3dgP8lN1cJu8EnTXnEh0iahbGCQGL0XoItBF4aWZEfZXzaSAhZicg/640?wx_fmt=jpeg&wxfrom=5&wx_lazy=1&wx_co=1)

150.《FAQ Retrieval using Query-Question Similarity and BERT-Based Query-Answer Relevance》

FAQ最近的一篇paper，思想上是完整的。

Query-Question：相似性建模

Query-Answer：相关性建模

模型融合：分数加权（一些基于长度的正则化技术）

149.《Neural Architectures for Named Entity Recognition》

LSTM+CRF的经典paper，里边还聊到一种transition-based alg,类似于依存分析的shift-reduced alg。（之前做把依存信号encode进bert的时候，第一次了解到shift-reduced的概念）

148.《Dialogue-Based Relation Extraction》

对话数据中做关系抽取的数据集。

147.《A Survey of Text Clustering Algorithms》,Charu C. Aggarwal, IBM

+ K-means(二分K-means、K-means++)和层次聚类是非常general的clustering算法
+ LDA+pLSA
+ Embedding-based(One-Hot, BOW, Word2Vec,TF-IDF,Sentence-BERT)
+ Online Clustering(较陌生)
+ 其他


146.《A Novel Hierarchical Binary Tagging Framework for Relational Triple Extraction》

![img146](https://github.com/zhpmatrix/zhpmatrix.github.io/blob/master/images/hbt.png?raw=true)


145.《PALM: Pre-training an Autoencoding&Autoregressive Language Model for Context-conditioned Generation》，ACL2020

![img145](https://github.com/zhpmatrix/zhpmatrix.github.io/blob/master/images/palm.png?raw=true)

144.PU Learning

《Learning Classifiers from Only Positive and Unlabeled Data》，2008

《An Evaluation of Two-Step Techniques for Positive-Unlabeled Learning in Text Classification》，2014

举一个简单的例子：训练二分类的时候，一般情况下，需要标注正样本和负样本；现在不了，只标注正样本就OK了，把剩下的当做负类（虽然剩下的可能包含正类）。一方面可以减小标注的难度，另一方面是对数据偏置的利用。整体上，可以放在半监督的视角下来看。


143.《Curate and Generate: A Corpus and Method for Joint Control of Semantics and Style in Neural NLG》

如何构造NLG的数据？

142.《Overestimation of Syntactic Representation in Neural Language Models》，ACL2020

140.《Natural Perturbation for Robust Question Answering》

139.《A Joint Model for Question Answering and Question Generation》

138.Query改写：《Learning to Paraphrase for Question Answering》，EMNLP2017

137.《Reading Wikipedia to Answer Open-Domain Questions》,Danqi Chen，应该还有不少玩儿法

136.《A Survey on Dialogue Systems: Recent Advances and New Frontiers》,JD的文章，简单清晰

135.吕正东两篇关于神经+符号的工作：

https://arxiv.org/pdf/1512.00965.pdf

用在对话系统，提升鲁棒性：

《Neural symbolic machines: Learning semantic parsers on freebase with weak supervision》

《Neural-symbolic machine learning for retrosynthesis and reaction prediction. Chemistry–A European Journal》

《Coupling Distributed and Symbolic Execution for Natural Language Queries》

《NEURAL SYMBOLIC READER: SCALABLE INTEGRATION OF DISTRIBUTED AND SYMBOLIC REPRESENTATIONS FOR READING COMPREHENSION》，ICLR2020（原来这个方向真的有一些工作在做。）

134.《From Machine Reading Comprehension to Dialogue State Tracking: Bridging the Gap》

核心方法：将DST任务用MRC的方式modeling，剩下的就是MRC的优点了。

133.《CrossWOZ: A Large-Scale Chinese Cross-Domain Task-Oriented Dialogue Dataset》，ACL2020

最近出的大规模，跨领域（酒店，餐馆等），任务型对话数据集。除了给出一个数据集之外，同时将一个对话系统分为五个部分：user simulator, nlu, dst, dpl, nlg.其中user simulator有特色。此外，在每个stage，都给出了一个实现。

补充：《MuTual: A Dataset for Multi-Turn Dialogue Reasoning》ACL2020

基于中国高考英语听力，需要逻辑和推理（BERT比人差很多）。难度比Ubuntu Dialogue Corpus高很多（已经被BERT拿下）。

132.最近的Review文章感觉比较多，梳理如下：

文本分类：《Deep Learning Based Text Classification: A Comprehensive Review》

关系抽取：《More Data, More Relations, More Context and More Openness: A Review and Outlook for Relation Extraction》

NER: 《A Survey on Deep Learning for Named Entity Recognition》

预训练模型：《Pre-trained Models for Natural Language Processing: A Survey》

知识图谱：《A Survey on Knowledge Graphs: Representation, Acquisition and Applications》

机器翻译：《Neural Machine Translation: Challenges, Progress and Future》

Chatbot：《Recent Advances and Challenges in Task-oriented Dialog System》

文本风格迁移：《Review of Text Style Transfer Based on Deep Learning》

实体链接：[《Neural Entity Linking: A Survey of Models based on Deep Learning》](https://arxiv.org/pdf/2006.00575.pdf)

面向切面的情感分析：《A Comprehensive Survey on Aspect Based Sentiment Analysis》

Low-Resource Setting: 《Low-resource Languages:A Review of Past Work and Future Challenges》

131.《Deep Learning Based Text Classification: A Comprehensive Review》,42页，最新综述，我看不动了...

130.《Poly-Encoders: Architectures And Pre-training Strategies For Fast and Accurate Multi-Sentence Scores》,ICLR2020

上篇文章的follower：《Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks》这篇文章，很早就浏览到，当时觉得不就是Encoder端换成BERT了吗？...不过看了ICLR2020的那篇，FAIR出品的，我都不清楚啥是好，啥是不好了？

最近做QA，重新翻到这两篇文章，有一些新的想法。

129.《Distantly Supervised Named Entity Recognition using Positive-Unlabeled Learning》

核心观点：只用词典做命名实体识别

128.《Coreference Resolution as Query-based Span Prediction》，ACL2020

李纪为第三篇用问答的思路做的工作，两个相关数据集的SOTA。第一步：找出候选实体；第二步：Query构建。包含候选实体的句子，用特殊标识符标识出候选实体。输入为Query+Context，输出为其他实体的位置。和之前的工作类似，一个样本仍旧需要多次inference完成训练和测试。

相关文章：

（1）《Dice Loss for Data-imbalanced NLP Tasks》

（2）《A Uniﬁed MRC Framework for Named Entity Recognition》

（3）《Scaling Up Open Tagging from Tens to Thousands: Comprehension Empowered Attribute Value Extraction from Product Title》

补充：

《Event Extraction by Answering (Almost) Natural Questions》，路子一致，用在事件抽取上很正常。

127.模型校准

logit就是置信度，就是概率？not for all.

概率校准解释的相对系统的文章：https://zhuanlan.zhihu.com/p/90479183

sklearn的介绍：https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration.html#sphx-glr-auto-examples-calibration-plot-calibration-py

核心观点：

+ 对于SVM，是基于margin的训练，没有prob

+ Platt Scaling:原始模型之外，重新训练一个模型(LR)

+ Isotonic Regression(保序回归)：曲线拟合

+ 概率校准方法的评估

关于预训练NLP模型的工作：《Calibration of Pre-trained Transformers》（个人围绕这块也做了一些工作）

《On Calibration of Modern Neural Networks》

《Training Confidence-Calibrated Classifiers For Detection Out-Of-Distribution Samples》,ICLR2018（还没读..）

《Distance-Based Learning from Errors for Confidence Calibration》ICLR2020（还没读..）

《Posterior Calibrated Training on Sentence Classification Tasks》ACL2020（还没读..）

126.阅读理解中的multi-span工作

启发：MRC也是一个经典的自然语言理解任务，同时包括很多子任务。在multi-span的问题设定下，有一些有意思的想法。

《Tag-based Multi-Span Extraction in Reading Comprehension》，BIO标注+**多种解码策略对比**

《A Multi-Type Multi-Span Network for Reading Comprehension that Requires Discrete Reasoning》

125.《Well-Read Students Learn Better: On the Importance of Pre-training Compact Models》

围绕BERT做模型压缩的工作，很扎实。附带Jacob Devlin最近在一个talk上的关于模型蒸馏的想法：

![img125](https://github.com/zhpmatrix/zhpmatrix.github.io/blob/master/images/model_distill.png?raw=true)


124.指代消解的论文（部门[大佬](https://github.com/jerrychen1990)的笔记）

End-to-end Neural Coreference Resolution:baseline版本，将预测span与预测两个span是否同指代放在一个任务里训练。**利用剪枝的方式避免过大复杂度的span pair搜索**

Higher-order Coreference Resolution with Coarse-to-ﬁne Inference：除了利用pair-wise的信息，利用整个cluster的信息做span matching

BERT for Coreference Resolution: Baselines and Analysis：在上一篇论文的基础上，利用bert代替原有的特征提取器

123.《Information Leakage in Embedding Models》

![img123](https://github.com/zhpmatrix/zhpmatrix.github.io/blob/master/images/Information%20Leakage%20in%20Embedding%20Models.png?raw=true)

122.《word2vec, node2vec, graph2vec, X2vec: Towards a Theory of Vector Embeddings of Structured Data》

121.《AliCoCo: Alibaba E-commerce Cognitive Concept Net》

阿里认知图谱的构建工作，PAKDD2020。

相关工作： 哈工大的大词林，相关的工作也是很赞。最近放出了很大一批[数据](https://mp.weixin.qq.com/s?__biz=MzIxMjAzNDY5Mg==&mid=2650797234&idx=1&sn=a2791160d643a04ace67e4630046b6b9&chksm=8f476159b830e84ff25c9a62b74a3074d8c8c4b57dfbec04faa689ee4be74e376ce900f6cee3&token=711471654&lang=zh_CN%23rd)。

120.《Improving Named Entity Recognition for Chinese Social Media with Word Segmentation Representation Learning》

Joint Learing： NER+CWS（应该也属于比较早的工作）

119.《CharNER: Character-Level Named Entity Recognition》

比较早期的文章了。证明了对于NER任务来说，不分词也是OK的。尤其对于中文和日文。

118.《Comprehensive Named Entity Recognition on CORD-19 with Distant or Weak Supervision》

这篇文章引用了一些文章，用词典加持NER任务，Jiongbo的好几篇文章。

117.不平衡问题的两篇paper

《Class-Balanced Loss Based on Effective Number of Samples》，CVPR2019

《Dice Loss for Data-imbalanced NLP Tasks》，NLP问题上（主要是NER和MRC任务上的不平衡），这篇文章针对NER和MRC任务中的标签不平衡问题，比较了包括ce，weighted ce，focal loss，dice loss等共计6种loss在多个理解任务上的效果。（我真的很少听说focal loss在一些任务上帮助很大，从2017年做一个比赛的时候就用，到2020年了，每次准备用的时候都要嘀咕一下，以后不再用了。）

116.《Pairwise Multi-Class Document Classification for Semantic Relations between Wikipedia Articles》

BERT系用于doc分类。

115. Multi-Hop MRC（这是一个有意思的问题）

《Cognitive Graph for Multi-Hop Reading Comprehension at Scale》，主要思想是：BERT+GNN。

《Building Dynamic Knowledge Graphs From Text Using Machine Reading Comprehension》，ICLR2019

问题的setting如下：

![115](https://github.com/zhpmatrix/zhpmatrix.github.io/blob/master/images/multi-hop-mrc.png?raw=true)

114.《One for All: Neural Joint Modeling of Entities and Events》

信息抽取领域的"宗教之争"：pipeline v.s. joint(二者各有优缺点，在自己的实践中，主要看个人的技术品味了，自己比较喜欢后者，但是还是要综合考虑数据，任务等各个方面的因素)

个人比较喜欢的工作：一个模型做事件抽取，包括trigger识别，实体识别，关系识别。（另外关于一个模型的做法，不仅在tagging方式上做，也可以通过共享encoder在classifier上做，inference的时候仍旧可以拆成多个模型，本质在于shared encoder和multi-task）

![114](https://github.com/zhpmatrix/zhpmatrix.github.io/blob/master/images/One%20for%20All_Neural%20Joint%20Modeling%20of%20Entities%20and%20Events.png?raw=true)

113.《Exploring Pre-trained Language Models for Event Extraction and Generation》

工作：pipeline+一种数据增强方案

![113](https://github.com/zhpmatrix/zhpmatrix.github.io/blob/master/images/Exploring%20Pre-trained%20Language%20Models%20for%20Event%20Extraction%20and%20Generation.png?raw=true)

112.《Felix: Flexible Text Editing Through Tagging and Insertion》

和EMNLP2019的这篇《Encode, Tag, Realize: High-Precision Text Editing》类似，上文的比较对象正是这篇文章。

111.《Prior Knowledge Driven Label Embedding for Slot Filling in Natural Language Understanding》俞凯老师组的文章

核心观点：融合先验知识到slot filling任务中，先验知识分为三类：

atomic concepts:原子概念，比如一个slot类型可以描述为其他多个子类型

slot description:就是slot类型的描述了

slot exemplars: 就是slot本身啦

整体上的方式就是确定要用的先验知识后，做encode，融合进slot filling任务中。前两种先验知识需要专家参与。

110.《Adaptive Name Entity Recognition under Highly Unbalanced Data》

提升robustness：关于noise和missing data

![img110](https://wx2.sinaimg.cn/mw690/aba7d18bly1gd5yvm5eyvj210y0ju489.jpg)

109.《AUC-Maximized Deep Convolutional Neural Fields For Sequence Labeling》,ICLR2016

《AUCpreD: proteome-level protein disorder prediction by AUC-maximized deep convolutional neural fields》，ECCB2016

同一篇文章。核心观点(用于sequence labeling任务)：

**The widely-used training methods, such as maximum-likelihood and maximum labelwise accuracy, do not work well on imbalanced data.by directly maximizing the empirical Area Under the ROC Curve (AUC), which is an unbiased measurement for imbalanced data.**


108.《Parallel sequence tagging for concept recognition》

实体识别和Linking都转化为一个labeling任务，这里Linking的目的是给一个实体编号，也就是基于Closed Set的假设。

107.《Contextual String Embeddings for Sequence Labeling》

CoNLL03的SOTA。

关于字和词的融合，也有很多玩法：

（1）直接融合分词信息

（2）先去学字的feature，然后基于词做融合，就是这篇了

（3）类似于ZEN，融合n-gram的信息

（4）......

![img107](https://wx3.sinaimg.cn/mw690/aba7d18bly1gd0r1ep530j214a0fg41q.jpg)

补充一篇相关文章，《Chinese NER Using Lattice LSTM》,ACL2018

想法：这篇工作和ZEN存在某种程度上的相关性，不过这篇的工作是基于LSTM特殊的网络结构来做的。而ZEN是在BERT的体系下完成的，比起分不分词，n-gram是一种更加灵活，同时具有信息量的表示。

![img072](https://wx2.sinaimg.cn/mw690/aba7d18bly1gd47nc552gj20bo069t9f.jpg)

看一个相关的，如何把专家知识融入实体识别？

![相关](https://pic4.zhimg.com/80/v2-52eb4dc87a89861438b0c04e2e98cda3_720w.jpg)

106.《Pre-trained Models for Natural Language Processing: A Survey》

黄萱菁，邱锡鹏老师们的工作。

相关工作：BERT的简易版Review，《A Primer in BERTology: What we know about how BERT works》

105.[《A Survey on Contextual Embeddings》](https://arxiv.org/pdf/2003.07278.pdf)

104.《Conditional BERT Contextual Augmentation》

刚刚看到一篇[文章](https://zhuanlan.zhihu.com/p/112877845)，关于用近期的一些技术做文本分类增强的。因此，很好奇就看了原始论文，其实类似的思想，在组里已经在很多地方用到了，简单写下笔记吧。

**核心：用label embedding替代segmentation embedding。**

为什么要做数据增强？（防止过拟合和提升泛化）

为啥NLP的增强不好搞？uncontrollable(semantic invariance, label correctness)

贡献是啥？

+ augment sentences without breaking the label-compatibility
+ can be applied to style transfer
+ SOTA 

相关工作？

+ sample-based methods
+ generation-based methods(gan, vae)
+ domain-specific methods(同义词替换之类)

103.《FixMatch: Simplifying Semi-Supervised Learning with
Consistency and Confidence》

利用weak/strong样本增强，实现semi-supervised的效果提升。近些年来，CV领域关于semi-supervised的一些相对不错的工作，整体上的特点就是简单，简单并不意味着容易想到。时至今日，越来越觉得，针对DL的问题，能够在“道路千万条”的前
