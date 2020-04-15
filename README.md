139.《A Joint Model for Question Answering and Question Generation》

138.Query改写：《Learning to Paraphrase for Question Answering》，EMNLP2017

137.《Reading Wikipedia to Answer Open-Domain Questions》,Danqi Chen，应该还有不少玩儿法

136.《A Survey on Dialogue Systems: Recent Advances and New Frontiers》,JD的文章，简单清晰

135.吕正东两篇关于神经+符号的工作：

https://arxiv.org/pdf/1512.00965.pdf

《Coupling Distributed and Symbolic Execution for Natural Language Queries》

134.《From Machine Reading Comprehension to Dialogue State Tracking: Bridging the Gap》

核心方法：将DST任务用MRC的方式modeling，剩下的就是MRC的优点了。

133.《CrossWOZ: A Large-Scale Chinese Cross-Domain Task-Oriented Dialogue Dataset》

最近出的大规模，跨领域（酒店，餐馆等），任务型对话数据集。除了给出一个数据集之外，同时将一个对话系统分为五个部分：user simulator, nlu, dst, dpl, nlg.其中user simulator有特色。此外，在每个stage，都给出了一个实现。

132.最近的Review文章感觉比较多，梳理如下：

文本分类：《Deep Learning Based Text Classification: A Comprehensive Review》

关系抽取：《More Data, More Relations, More Context and More Openness: A Review and Outlook for Relation Extraction》

NER: 《A Survey on Deep Learning for Named Entity Recognition》

预训练模型：《Pre-trained Models for Natural Language Processing: A Survey》

知识图谱：《A Survey on Knowledge Graphs: Representation, Acquisition and Applications》

机器翻译：《Neural Machine Translation: Challenges, Progress and Future》

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

利用weak/strong样本增强，实现semi-supervised的效果提升。近些年来，CV领域关于semi-supervised的一些相对不错的工作，整体上的特点就是简单，简单并不意味着容易想到。时至今日，越来越觉得，针对DL的问题，能够在“道路千万条”的前提下，找到最合适的那几条，做对几件事，基本就可以带来很大的提升。过于依赖试错，导致经验这种东西也显得很飘。如果能够给经验一个定量描述，那就再好不过了，不过似乎目前还没有发现，包括自己也是。能够做的是，还是要结合场景和分析大量的badcase，选择要做的事情。很多人脱离badcase，空谈general的优化，导致试错空间巨大，成本巨高，实在太过于粗放。

![img103](https://wx1.sinaimg.cn/mw690/aba7d18bly1gctdqyep7oj20q50anq64.jpg)


102.《A Hierarchical Multi-task Approach for Learning Embeddings from Semantic Tasks》

总结了很多multi-task训练的Trick。

![img102](https://wx3.sinaimg.cn/mw690/aba7d18bly1gcsa3ifkfjj20lr0o3mzk.jpg)

101.《Speeding Up Neural Machine Translation Decoding by Cube Pruning》

一种decode端的技术。

100.《Sentence-Level Fluency Evaluation》

no-reference evaluation.

99.《CASE: Context-Aware Semantic Expansion》

暴力的方法，将问题转化为一个多标签分类问题（所有下位词的集合）。为了解决分类总数较多的问题，采用了特殊Trick解决。

98.《Improving Neural Named Entity Recognition with Gazetteers》

围绕NER，最近关注的几个Trick：

（1）LAN用于替代CRF

（2）ZEN，融合n-gram的信息

（3）这篇文章，with Gazetteers.（**TIPS：《现代汉语词典》也是一个高度结构化的好东西，为啥没人用呢？**）

![98](https://github.com/zhpmatrix/zhpmatrix.github.io/blob/master/images/Improving%20Neural%20Named%20Entity%20Recognition%20with%20Gazetteers.png?raw=true)

97.如何利用语言学提升任务表现？

一个朴素的观点：数据不够，先验来凑。啥是先验？比如语言学。知识图谱也可以认为是先验的一种载体，但是更多的承担的是common sense/world knowledge的角色.

[《Why BERT Fails in Commercial Environments》](https://www.intel.ai/bert-commercial-environments/#gs.ykp1xd)

[《Attending to Entities for Better Text Understanding》](https://arxiv.org/abs/1911.04361)

[《Linguistically-Informed Self-Attention for Semantic Role Labeling》](https://arxiv.org/pdf/1804.08199.pdf)

**语言学信号融合**

关键问题：(1)如何量化语言学信号？(2)如何对多种语言学信号融合？

词性信号：词性标注集[参考](http://www.ltp-cloud.com/intro#pos_how)，语言学信号暂且来自LTP，不做分词策略的对齐(LTP和WordPiece)。量化思路：直接表示；间接表示(BERT的hidden representation)；

依存句法信号：依存句法关系[参考](http://ltp.ai/docs/appendix.html#id5)，量化思路：直接表示；转化为邻接矩阵；Graph Embedding；(@凡哥@松旭)

语义角色信号：非对齐输出，后续考虑如何加入。

初步结论：指标下降。猜测原因之一是添加了新的embedding需要去learn，当前训练集不足以learn到质量较高的embedding，在一定程度上意味着在信号融合端会造成noise的引入。

参考文章：

+ 《Linguistic Input Features Improve Neural Machine Translation》

+ 《Extending Neural Question Answering with Linguistic Input Features》

+ 《Semantics-aware BERT for Language Understanding》


96.《Data Augmentation using Pre-trained Transformer Models》

文章比较了基于自编码(BERT)，预训练语言模型(GPT-2)，和预训练seq2seq(BART/T5)的三种模型用于数据增强的效果。具体的，比如对于一个情感分类任务，三种方式都可以做，哪种好一些？文章的结论是：考虑到标签保持的能力和多样性，seq2seq整体上较好。
在具体数据增强的方法上，文章也有一些阐述。整体上文章解决的问题实用性较强，既可以作为一篇Review，也可以作为一篇技术报告来看。此外，文中的一些方法虽然是放在数据增强的角度来考察的，但是理论上应该也可以推广到其他Task上，例如情感迁移等。

[《Learning from Unlabeled Data》by Thang Luong](https://drive.google.com/file/d/1ax1-XprJHDRRv2Ru3dJwPLs3ShxcpQ3r/view)

这个Talk主要讲了：UDA+NoisyStudent(self-training)+Meena（chatbot）

95.《Train Large, Then Compress: Rethinking Model Size for Efficient Training and Inference of Transformers》

讨论了一个比较实际的问题，如下：

![img95](https://bair.berkeley.edu/static/blog/compress/Flowchart.png)

94.关于CensorShip的三篇文章：

《Creative Language Encoding under Censorship》

《Linguistic Fingerprints of Internet Censorship: the Case of Sina Weibo》

《How Censorship in China Allows Government Criticism but Silences Collective Expression》

93.《A Primer in BERTology: What we know about how BERT works》

姑且算是一篇关于BERT的Review吧，后面一定有质量更高的Review。不过这篇文章的特色主要在于引用了更多关于probing的工作。由于probing工作，在任务设计的时候，可能就已经引入了过多的假设，导致结论理论上带有一定的bias。总体上，仍旧值得读一下。

92.《Rethinking Bias-Variance Trade-off for Generalization of Neural Networks》,马毅老师的工作

期望风险曲线 = bias曲线+variance曲线，当bias曲线和variance曲线的scale不同时，期望风险曲线就会呈现出不同的形态。类比高中物理学波运动和分解...反正对马老师说，这篇文章就是homework而已。如下：

![img92](https://wx2.sinaimg.cn/mw690/aba7d18bly1gcdmxi04loj20xl097djl.jpg)

91.《A Generalized Framework of Sequence Generation with Application to Undirected Sequence Models》

90.《End-to-End Entity Linking and Disambiguation leveraging Word and Knowledge Graph Embeddings》

![img90](https://wx2.sinaimg.cn/mw690/aba7d18bly1gcc0x4g7fsj20k90lzq96.jpg)

89.《Semantic Relatedness for Keyword Disambiguation: Exploiting Different Embeddings》

这篇文章主要解决的问题是：

> traditional WSD techniques do not perform well in situations in which the available linguistic information is very scarce, such as the case of keyword-based queries.

WSD任务的入门介绍：https://blog.csdn.net/jclian91/article/details/90117189

88.《Label-guided Learning for Text Classiﬁcation》

水文，不议。

87.《Fast Structured Decoding for Sequence Models》

非自回归Transformer和自回归Transformer的区别，主要的贡献在Decoder的改造，类CRF，该技术不但可以用在seq2seq上，同时也可以用在字到字的纠错任务上，否则原生的CRF，你train个试一试？字典大的吓屎人。

![img87](https://wx4.sinaimg.cn/mw690/aba7d18bly1gc7sjgw4m3j20r60g1gpi.jpg)

86.《Joint Embedding in Named Entity Linking on Sentence Level》

85.《Neural Grammatical Error Correction Systems with Unsupervised Pre-training on Synthetic Data》

比较新的SOTA，seq2seq做gec任务。

84.《An Empirical Study of Incorporating Pseudo Data into Grammatical Error Correction》

围绕seq2seq的思路做gec任务，做了很多实验性的对比，包括：数据构造策略，后处理，原始数据使用等。最终的一个经验性结论是：Gigaword作为种子数据源+BackTranslation+seq2seq在经典benchmark上可以取得不错的结果（看了下Precision和Recall，均不到0.75！）。

不过，整体上提供了很多可以尝试的idea。

83.《LAMBERT: Layout-Aware language Modeling using BERT for
information extraction》

抽取PDF时，可以融合Layout的信息。印象中达摩院在PDF抽取中也有类似想法的工作，[相关比赛](https://tianchi.aliyun.com/competition/entrance/231771/information)

82.《A New Clustering neural network for Chinese word segmentation》

Clustering的思路做分词。先将问题转化为一个multi-label的问题，然后做Clustering(GMM/K-means)

81.《Identifying Relations for Open Information Extraction》，引用量1000+

给出了一种开放信息抽取的方式，总结了漏报和误报的pattern，同时给出了一种基于feature构建的三元组质检的方式。

80.《Fine-Tuning Pretrained Language Models:
Weight Initializations, Data Orders, and Early Stopping》

随机种子对最终结果的影响是很大的。这篇文章探讨了随机种子对权重初始化和训练数据排序的影响。

TIPS：心情不好，换个种子？

79.《Incorporating BERT into Neural Machine Translation》,ICLR2020

结论：7个数据集的SOTA

思路：BERT只负责抽feature，然后和seq2seq的encoder&decoder的layers的feature做融合。

示例图如下：

 ![img79](https://wx1.sinaimg.cn/mw690/aba7d18bly1gc1i5zzevqj20ii0c5wgn.jpg)
 
 这个方向很多人在探索，比如现在相对有效的是，encoder端用BERT做fine-tune，decoder端train from scratch.


78.《How Contextual are Contextualized Word Representations?》,EMNLP2019

对context的一种量化分析方法。一个概念是否是玄学，与定义和度量的方式有一定关系。

77.《Utilizing BERT Intermediate Layers for Aspect Based Sentiment Analysis
and Natural Language Inference》

![img77](https://wx4.sinaimg.cn/mw690/aba7d18bly1gbvtwwpwocj20er0dowg7.jpg)


76.《Encode, Tag, Realize: High-Precision Text Editing》,EMNLP2019

一大早，我老大发的一篇Paper。

一张图说明，如下：

![img76](https://wx4.sinaimg.cn/mw690/aba7d18bly1gbvqpq61e9j20wd04ijsg.jpg)

适合解决的问题：生成类任务中，输入和输出overlap较多的时候。传统的seq2seq在解决这类问题时，就显得效率会很低，主要是由于decoder端的token是都要生成得到。

主要内容：Encode端是BERT, Tag表示针对特定任务的操作(删除，保留，添加等)，Realize是针对Tag序列的聚合阶段。整体上是一个基于BERT的序列标注任务，特色在于针对特定任务设计特殊的Tag体系，从而完成任务目标，如文本融合，压缩，纠错等。

优点：典型任务上的SOTA(虽然有些任务是微弱提升)，推理速度，方法体系相对简单。

缺点：Tag阶段做不好，会直接影响到最终效果。当Tag阶段得到的Vocab比较大的时候，导致模型的分类层比较重，或许可以采用分级分类的想法来缓解。

其他：在纠错任务上，作为文章的对比对象，《Neural Grammatical Error Correction Systems with Unsupervised Pre-training on Synthetic Data》的F值比该工作高了近20个百分点，文章认为主要的原因是，前者使用了更多外部数据。

[PR文档](https://www.jiqizhixin.com/articles/2020-02-10-8)，文档中的内容和Paper不完全一致，可以直接看Paper。

75.《A review of novelty detection》，Marco A.F. Pimentel 2014年，310篇引文

74.《Snippext: Semi-supervised Opinion Mining with Augmented Data》，WWW2020

属于BERT系工作，Snippext = MixDA（一种数据增强技术）+MixMatch（一种半监督学习算法）。文章中提到，这套方法已经用在了酒店评论聚合平台和某猎头公司。

73.《Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift》，NeurIPS2019

稍后会写一篇博客总结这类问题，与OOD相关。

![img73_1](https://wx2.sinaimg.cn/mw690/aba7d18bly1gbsukjkw9qj20z2072tbv.jpg)

![img73_2](https://wx2.sinaimg.cn/mw690/aba7d18bly1gbsukfeyquj20zc0dvtdt.jpg)

![img73_3](https://wx4.sinaimg.cn/mw690/aba7d18bly1gbsukaki2wj20z50a1adu.jpg)

72.《Deep k-Nearest Neighbors: Towards Conﬁdent, Interpretable and Robust Deep Learning》

> This hybrid classiﬁer combines the k-nearest neighbors algorithm with representations of the data learned by each layer of the DNN: a test input is compared to its neighboring training points according to the distance that separates them in the representations.

NN是一个非常general的方法，印象中2018年的CVPR有篇关于detection的文章，此外，在之前做的一些比赛中，可以作为一个post-processing的手段。一直在思考的问题是：能否作为一种解决长尾分类的方法？

71.《Concept Discovery through Information Extraction in Restaurant Domain》

word2vec+cluster,类似工作，类似方法，效果一般不差。

70.《Editable Neural Networks》，ICLR2020，Poster

这篇文章是最近看到的非常有启发的文章，体现在以下几点：

（1）这篇文章的方案具有工业实用性。针对badcase的hotfix手段。

（2）针对hotfix的问题，也正是lifelong learning可以发挥能力的地方，只不过这里是few-shot的sample。

（3）baseline中关于KNN的部分也带来了一些启发。长尾数据和badcase的fix问题。

在这篇文章的[OpenReview](https://openreview.net/forum?id=HJedXaEtvS)中，其中一个Reviewer提出了一个有意思的问题，该问题其实是OOD相关的，同样是一个有趣且非常重要的问题，如下：

![img70](https://wx4.sinaimg.cn/mw690/aba7d18bly1gbpcvlvgzhj20sz02lab0.jpg)

69.《Lifelong Learning for Sentiment Classification》

Bayes用于LLL，给出了LL的一个定义，需要四个组件。

68.《Deep learning for drug–drug interaction extraction from the literature: a review》

下载不到，但是想看。

67.《The Power of Pivoting for Exact Clique Counting》，WSDM2020 Best Paper

私以为相同类型的工作：围绕最近邻相关的工作，以及本篇文章。

《A holistic lexicon-based approach to opinion mining》WSDM2008，WSDM2020 Test of Time Award

66.《Graph Convolution for Multimodal Information Extraction from Visually Rich Documents》 ，达摩院

一般的做法是：半结构化数据到文本，比如PDF转化为文本，然后对文本做抽取。这里说不仅要文本feature，同时要融合visual feature（GCN获取）做sequence labeling。

问题：既然是半结构化数据，能否只利用visual feature呢？

[相关比赛](https://tianchi.aliyun.com/competition/entrance/231771/rankingList)(从人物PDF简历中抽取人名，学历，年龄等信息）

65.《Empower Sequence Labeling with Task-Aware Neural Language Model》

multi-task用于优化sequence labeling，LM-LSTM-CRF用于sequence labeling，围绕直接的LSTM/BERT做序列标注，可以用LM/LAN/CRF优化。

64.《Rethinking Generalization of Neural Models: A Named Entity Recognition Case Study》

从NER任务谈神经网络模型的泛化性。定义了一些比较浅的指标，找出了公有数据集的一些bug。看似简单，但是这些问题实际场景下会经常遇到。

63.《A Baseline For Detecting MisClassified And Out Of Distribution Examples In Neural Networks》，ICLR2017

提出一个很有意思的问题：针对二分类问题，当你的模型预测一个样本prob很低的时候，这个样本是负样本，还是这个样本是OOD？（out of distribution）

最近做的一个序列标注的工作中观察到：OOD一般的prob确实要低很多。从置信度的角度来理解，make sense。

该方向上的工作与鲁棒性强相关。

62.《Q8BERT: Quantized 8Bit BERT》,NeurIPS2019

模型压缩和加速的Topic。

相关Paper：《Quantizing deep convolutional networks for efﬁcient inference: A whitepaper》

PyTorch1.3.0开始支持这个feature，不过目前是experimental的，[tutorial地址](https://pytorch.org/tutorials/)

transformers的issues区有人提出了相关[issue](https://github.com/huggingface/transformers/issues/2466)

61.《PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization》

提示：共52页，附录40页。方法如下：

![img](https://wx1.sinaimg.cn/mw690/aba7d18bly1gatvxcslymj20oj0bgwgu.jpg)

60.《Attention Is All You Need》（Transformer的概念难道不是在这里定义的吗？为啥好多paper瞎写，好多人瞎说，不是只有encoder啊。）

亮点：

（1）SOTA
（2）并行(feed forward是position wise的；self-attention的矩阵乘法计算；multi-head等)
（3）简单（抛弃了recurrence和conv）

+ Attention: 在建模input和output之间的dependency时，无需考虑二者之间的distance。dependency不一定必须是distance的函数，取决于如何定义dependency?

描述： output = f(a query, a set of key-value pairs)

为啥选择dot-product attention而不是additive attention？**最主要的目的还是工程上快。因为存在计算更快，空间更加高效的矩阵乘法。**二者的理论时间复杂度一样，实际上，后者的效果不依赖于待计算vector的维度大小，且整体上效果更好。但是还想用前者，那就必须消除vector的维度影响，因此分母除sqrt(d\_k)。实际上，对前者，V的weight是直接计算得到，而对后者，是learn到的。那么，为啥d\_k会对前者产生影响？主要是qk的variance是与d\_k有关的，当d\_k很大时，variance很大，这样容易导致gradient进入softmax的饱和区，玩儿个锤子。

Transformer的优点？（思考角度）

（1）每层的计算复杂度

（2）可并行的总量

（3）能否很好地learn到long range dependency（很遗憾，这里dependency仍旧是distance的函数）

（4）可解释性

+ 时间复杂度：关联input和output任意两个positon的信号，需要的操作的次数？Transformer=O(constant),ConvS2S=O(N),ByteNet=O(logN,不是很确定),这里的区别和用数组还是用链表相似。

+ Memory Network是基于recurrent attention机制的，不是sequence-aligned recurrence（类似2014年经典的seq2seq+attention），从这点儿来讲，Transformer也算是延续了Memory Network的血脉，尤记得当年Memory Network🔥过。

+ position encoding

两种方式如下：

（1）fixed（相对编码）

（2）learnable

结论：效果差不多。但是learnable的需要预先learn到才能用（**BERT，长度512**），但是fixed版就是pos的函数，可以直接计算，因此可以扩展到任意input长度序列。

+ Transformer定义:

>  The Transformer follows encoder-decoder structure using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder.

Decoder端的三个细节：

+ Masked Multi-Head Attention

对于“我 爱 吃 苹 果 。”， 预测“吃”这个token的时候，不能看到“苹果。”这三个token。

+ shifted right

训练时：输入“<sos> 我 爱 吃 苹 果 。”， 预测“我 爱 吃 苹 果 。<eos>”

预测时（自回归）：生产者-消费者模型，一次一个。

+ 共享

encoder端和decoder端的embedding层共享，pre-softmax linear层共享。这里有意思的点儿是，**将一些机制用于预训练seq2seq模型中？**

三种Regularization策略：

（1）每个sub-layer的输出

（2）sums of the embeddings

（3）label smoothing:虽然会hurt到ppl（该trick的目的就是使得模型变得unsure），但是可以提高bleu。

其他Trick：

（1）average last few checkpoints

（2）基于devset，找到beam size值和length penalty值

（3）maximum output length = input length + 50，允许早停

59.《GPT-based Generation for Classical Chinese Poetry》

整体思路如下：

![img59](https://wx1.sinaimg.cn/mw690/aba7d18bly1gana6e98duj20r30g140q.jpg)

该思路可以对比中文纠错的两种思路：

（1）seq2seq

（2）序列标注（自编码）

对于对联，诗等的生成，（1）是经典思路，（2）是本文的思路（自回归）。用序列标注的优点之一在于：平行数据是天然对齐的。此外，两个问题：

（1）自编码可行？（针对对齐场景）

（2）中文维基的预训练 vs 古诗词的预训练（两种语言结构）

最后，看到文章写道：

> Though the generated poems are not perfect all the time, our preliminary experiments have shown that GPT provides a good start to promote the overall quality of generated poems.

心头一惊。


58.《Looking Beyond Label Noise: Shifted Label Distribution Matters in Distantly Supervised Relation Extraction》

没有认真读，这里藏着，主要讨论：一个经典关系抽取数据集中的数据bias对metric的影响。

类似工作：[人脸识别数据集的身份重合问题](https://zhuanlan.zhihu.com/p/31968327)

其他还有一些，记不起来了。

57.两个有意思的经验性观点：

+ 《Deep Double Descent: Where Bigger Models and More Data Hurt》

> By considering larger function classes, which contain more candidate predictors compatible with the data, we are able to ﬁnd interpolating functions that have smaller norm and are thus “simpler”. Thus increasing function class capacity improves performance of classiﬁers.

参考如下图：

![57_img](https://wx1.sinaimg.cn/mw690/aba7d18bly1gaih6g6q7kj20tz0dywgw.jpg)

+ 《The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks》

彩票假设：

> A randomly initialized, dense neural network contains a subnetwork that is initialized such that when trained in isolation, it can match the test accuracy of the original network after training for at most the same number of iterations.

彩票假设的启发：

（1）pruning irrelevant weight

（2）retrain from scratch using only the 'lottery ticket' weights

56.《K-BERT: Enabling Language Representation with Knowledge Graph》,AAAI2020

融合KG的信息，啥也不想说了。

55.《PRETRAINED ENCYCLOPEDIA: WEAKLY SUPERVISED KNOWLEDGE-P RETRAINED LANGUAGE MODEL》,iclr2020

这篇文章通过设计预训练任务，希望guide模型去learn到更多的fact。整体思路是：对于一句话，找到话中的实体，用其他类型相同，但是内容不同的实体替代，然后训练模型，训练目标是多任务形式，实体有没有被替换+MLM。至于这样做的原因，文章中提到：

> Compared to the language modeling objective, entity replacement is deﬁned at the entity level and introduces stronger negative signals. When we enforce entities to be of the same type, we pre- serve the linguistic correctness of the original sentence while the system needs to learn to perform judgment based on the factual aspect of the sentence.

类似思想其实可以用在很多其他地方。

相关文章：《Language Models as Knowledge Bases?》

54.《Probing sentence embeddings for linguistic properties》，ACL2018

属于Probing类工作，讨论句子embedding和语言学属性之间的关系。设计的三个层面的probing任务值得关注一下（但是个人并不认为这些能够完整体现语言学属性。另一方面，此类任务的设计似乎并不是一个简单的任务）：

+ surface information:句子长度等

+ syntactic information:语法树的深度等

+ semantic information: 词的时态

53.《Learning Sparse Sharing: Architectures for Multiple Tasks》, AAAI2020

类似思想体现在很多地方。文章中的一张图有意思：

![img53](https://mmbiz.qpic.cn/mmbiz_png/vJe7ErxcLmgUZb0kT1OmH0CNKpqTd77uMiazADJHiauicTMoia6LmkjtZNDUic1GF1vnU9syIYzibKicUSej07cM60p2Q/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)


52.《Multi-channel Reverse Dictionary Model》,AAAI2020

解决的问题：给定对词的描述，返回对应的词。

整体思路：对词的描述进行encode，然后和词的embedding算距离。因此，需要好的对词的描述进行特征抽取的encoder。

文章工作：加feature。（对于multi-channel相关的，多是加feature）

想法：不谈novelty，解决的问题很有意思。对于文章对应的系统，可以爬一爬，做weak-supervision相关。

51.《Likelihood Ratios for Out-of-Distribution Detection》

略读。问题：对OOD样本的处理。

想法：训练模型时，符合IID的假设；但是当模型部署到线上的时候，有时候很难保证IID能够满足，首先**不知道什么时候会出现OOD**，比如测试数据会随着时间变化。**如果出现OOD，这个时候如何处理OOD呢？**

（1）对OOD赋予一个较低的置信度，模型对待OOD的样本要保守。但是，有时候模型不会这样表现，这又是另外一个有趣的问题了；

（2）给定一个样本，判断是否是OOD?这个可以做二分类，可以训练一个生成模型，可以转化为一个outlier detection任务等；

NeurIPS2019，Bengio单独讲了一节，从IID->OOD，希望之后的工作从1-型系统转向2-型系统，也就是从感知层转向推理层，推理一定程度上是彻底解决OOD的方法之一。

50.Contrastive Learning

《Representation Learning with Contrastive Predictive Coding》，deepmind，提出了一个统一的框架。

《DATA-EFFICIENT IMAGE RECOGNITION WITH CONTRASTIVE PREDICTIVE CODING》，deepmind，这篇是上篇主要在image中的应用，做了一些改变，更偏应用。

《Momentum Contrast for Unsupervised Visual Representation Learning》，将contrastive learning的想法用于image领域的无监督预训练。

49.《DATA-DEPENDENT GAUSSIAN PRIOR OBJECTIVE FOR LANGUAGE GENERATION》，iclr2020

主要工作：在传统文本生成的loss函数（mle）中，添加一个kl term，这个term是基于gaussian，data-dependent的，主要动机是考虑那些非max prob的prob的分布，这些prob之间的信息同样是非常丰富的哦。举个简单例子：

猫，猪，苹果三分类任务中，对猫的预测结果是（0.7，0.2，0.1），分别对应猫，猪，苹果，这个预测是正确的，但是（0.2，0.1）也是有信息结果的，这个prob向量意味着猫和猪的距离比猫和苹果的距离近啊。

想到了什么？知识蒸馏呗，同样包括一些针对点，用一个分布来modeling的一些工作，好像在一些regression模型中有用。怎么挖掘这些信息，用一个分布来表达，这里的问题是：用啥分布？怎么得到分布的参数？（learnable）

除此之外，该工作融合了prior的信息，实验证明，low resource的setting下，提升明显；但是数据量大的时候，效果变弱。不管怎样，将prior信息融合model，是一个有趣的方向，可做的事情应该挺多的。

整体上，文章的实验，分析和动机都齐全，novelty倒也不是特别大，但是仍旧是一篇好文。东南大学的耿鑫老师做的label distribution learning其实也挺有意思，不过没有深挖。

48.《HOW MUCH POSITION INFORMATION DO CONVOLUTIONAL NEURAL NETWORKS ENCODE ?》，iclr2020

结论：**surprising degree of absolute position information that is encoded in commonly used neural networks.**

47.《Sequence Modeling with Unconstrained Generation Order》,NeurIPS 2019

seq2seq中，decoder不一定都要按照left-to-right的顺序，也可以任意方向。主要的工作集中在：

（1）允许任意方向：去掉mask

（2）如何定义方向？用position

（3）如何learn？token+position同时learn

有点反直觉？对啦。来来来，先写第三个字，再写第二个字，再写第八个字...无论怎样，从理论上讲，exposure bias应该会缓解。开个脑洞：基于bert，用sequence lableling的思路来做生成，解决output端和input端不需要严格对齐的问题。总之，memory bank也不要了吧。

46.《Mixtape：Breaking the Softmax Bottleneck Efficiently》，NeurIPS 2019

基于一个假设：**Softmax瓶颈就是说模型的低秩性无法充分表达高秩语言空间**，衍生出的两个方法Mos和Mixtape。其中Mos是指多个softmax的混合, Mixtape在效果上和Mos基本一致，但是大大提升了速度，因为本质上在保证高秩的同时，相比于Mos需要计算多次softmax，同时保存中间logit，在存储和速度上都有改善。

需要思考的点儿：

（1）假设本身的合理性？

（2）“信息冗余”的思想在一个侧面上的多种实现？

类似工作：各种各样的weight normalization实现......逃。

45.《“爱情像数学一样复杂”：用于社交聊天机器人的比喻生成系统》

最近比较关注word2vec的更多的应用，相关工作包括这篇文章，大词林的工作，**Trans系列（还没系统想过）**。

这篇文章主要解决给定一个本体，生成一个基于本体的比喻句。整体分为三个工作：

（1）找喻体

（2）找连接词

（3）套模版

这件事情能做的本质原因：本体往往来自抽象域，而喻体往往来自具体域。换言之，人们设法利用比喻，用容易理解的具体概念（即喻体）来解释和表达不易理解的抽象概念（即本体）。（通俗点说：你咋说都是对的。其实，古诗词也是这样滴，古诗词的韵味在于诗词本身和读者之间的交互，具体比如通过联renwu想的方式等）
33.《Image Classiﬁcation with Deep Learning in the Presence of Noisy Labels: A Survey》

44.《NEZHA: Neural Contextualized Representation for Chinese Language Understanding》,[相关介绍](https://mp.weixin.qq.com/s/RkCLSRyy_GuLOXMVSzTMdA)

改进维度：

模型：相对位置编码的利用（Functional Relative Positional Encoding，不需要learn。）

预训练任务：词级Mask+Span信息的利用

训练算法：混合精度+适用于大Batch(300000)的优化器LAMB

在训练过程中，我们采用混合精度训练（Mixed Precision Training）方式，在传统的深度学习训练过程中，所有的变量包括weight，activation和gradient都是用FP32（单精度浮点数）来表示。而在混合精度训练过程中，每一个step会为模型的所有weight维护一个FP32的copy，称为Master  Weights，在做前向和后向传播过程中，Master Weights会转换成FP16（半精度浮点数）格式，权重，激活函数和梯度都是用FP16进行表示，最后梯度会转换成FP32格式去更新Master Weights。

Ablation Study: 位置编码，masking策略，span预测任务，训练序列的长度，训练语料的大小均能带来提升，其中位置编码会带来显著提升；

**最大的收获：位置编码的改进。**

43.《DEEP ENSEMBLES:A LOSS LANDSCAPE PERSPECTIVE》

**Why do ensembles trained with just random initialization work so well in practice?**

42.《Learning From Positive and Unlabeled Data: A Survey》

相关：《A Survey on Postive and Unlabelled Learning》

41.《Distantly Supervised Named Entity Recognition using Positive-Unlabeled Learning》

40.《150 Successful Machine Learning Models: 6 Lessons Learned at Booking.com》,KDD2019, Applied Data Science Track

从Booking.com发布的150个机器学习模型，总结了6条经验，并提供了一些case分析。比如：提高模型的性能并不一定会转化为业务价值的增长；一些最强大的改进并非是在给定setting的上下文中改进模型，而是更改setting本身；减少预测服务的延迟（模型冗余/稀疏模型/预计算和缓存/请求打包/尽可能少的特征变换），尽早获得关于模型质量的反馈（响应分布分析）等

相关文章：

（1）[《Machine Learning:The High-Interest Credit Card of Technical Debt》](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/43146.pdf)

（2）[《Hidden Technical Debt in Machine Learning Systems》](https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems.pdf)

（3）[Github: Production Level Deep Learning](https://github.com/alirezadir/Production-Level-Deep-Learning)

39.《Open Domain Web Keyphrase Extraction Beyond Language Modeling》，EMNLP2019

开放域的关键短语抽取。主要贡献：

（1）做了一个目前可能最大的开放域的英文短语抽取数据集。

（2）**提出基于网页搜索的弱监督方案。**（目前的网页搜索，是基于纯字符串匹配，作为候选页面，很早自己就有这个想法，其实想办法用起来还是很酷的。）

（3）提出利用visual feature做增强，可能和数据集构建有关吧，个人不是很感兴趣。

（4）做了一个有点丑陋的架构（基于预训练语言模型）。

难点：当要去learn一个非常general的概念的时候，可能是hard to learn的。

38.《Generating Abstractive Summaries with Finetuned Language Models》

Alexander M. Rush组的工作，讨论预训练语言模型用于文本生成，特别是在seq2seq架构下。目前有很多人在关注这个方向，包括自己组里也在做一些探索，不过效果不是特别好。这篇文章，大致是decoder端用了一个预训练语言模型，不过取得的结果<=transformer+copy在摘要任务上的效果。

37.《TENER: Adapting Transformer Encoder for Named Entity Recognition》

魔改模型上有启发。文章不和BERT比，其实感觉对比意义也不是很大。如果基于此结构，做pretrain，会不会进一步提升BERT在NER任务上的表现？虽然文章讲了Transformer在NER任务上表现不好的原因，但是很好奇是否是真的不好？（这个需要更多实验结果的支持。在一个善于讲故事的年代，总得发生点什么才显得可信。）

36.《A Deep Look into Neural Ranking Models for Information Retrieval》

一篇关于Ranking的综述文章。hinge loss可以用在pairwise的ltr任务中。

35.《PyTorch: An Imperative Style, High-Performance Deep Learning Library》，NIPS2019

讨论系统哲学的学术文章，其实官方博客已经讨论了很多相关思想了，所以这篇作为学术文章可能就是方便大家引用吧。作为宇宙最屌框架，PyTorch已经不需要用一篇文章的引用量来证明自己了。

34.《Using Knowledge Graphs for Fact-Aware Language Modeling》，ACL2019

相关的想法：

（1）bert系通过mlm能否直接fact-aware？已经有人做了一些探索。

（2）没有没关系。把kg的信息融入encoder端，提升decoder的能力，就是这篇了。

（3）结合各个任务，融合kg的信息，那么怎么融合kg的信息，就是一个可以玩儿出花儿来的事情。

33.《Spam Review Detection with Graph Convolutional Networks》，CIKM2019最佳应用论文

将GCN用于闲鱼的垃圾营销广告检测。特征提取层上分为两个部分，分别是基于商品-用户-评论的二部图和不同评论构成的图，GCN作为特征提取器。分类层采用TextCNN。

32.《Chinese Word Segmentation as Character Tagging》，2003年的文章，引用量很高。

好像是第一次将分词问题modeling为一个tagging问题来做，共有四个标签（LL/MM/RR/LR）如下：

![img_word_segmentation](https://wx3.sinaimg.cn/mw690/aba7d18bly1g9cvr16rhwj20t808t40d.jpg)

31.《How to Ask Better Questions? A Large-Scale Multi-Domain Dataset for Rewriting Ill-Formed Questions》

做了一个问题改写的数据集。

30.《Learning Semantic Hierarchies via Word Embeddings》，ACL2014

《大词林》，用word2vec的思路做“上下位”关系挖掘。word2vec用好了，应该可以解决很多问题，包括embedding，相关性等任务。

用于开放域的实体类别层次化（上下位关系）。核心思想是用词向量之间的差值刻画上下位关系。工作分为三个步骤：词向量学习，映射矩阵学习和上下位关系判断。上下位关系的判断也可归于关系抽取任务，是一种特殊的较为抽象的关系。因此可以很自然地将文章的想法拓展到关系分类任务上，对每类关系学习一个映射矩阵。当用于SPO三元组直接抽取时，需要有针对性的映射关系学习方法。

29.《Statistical Machine Translation: IBM Models 1 and 2》，Michael Collins

Noisy Channel Model的经典案例：作为统计模型用于机器翻译任务。除此之外，可以用于拼写纠错，Auto Suggestion等，大二时实现的Bayes Matting也有类似的感觉。总之，NCM是一个比较general的理论模型。

28.《Few-Shot Sequence Labeling with Label Dependency Transfer and Pair-wise Embedding》

 Few-Shot Learning的工作，用于命令实体识别任务上。

27.《Hierarchically-Reﬁned Label Attention Network for Sequence Labeling》,EMNLP2019

Label Embedding+Attention用于sequence labeling。平均提升了不足一个百分点，可能和相关任务原来的指标已经较高有关系。不过工作做的还是比较干净的。

文章的主要贡献：精度略有提升的前提下，速度提升了不少。主要的方式是基于label probability distribution，做self-attention。从效果来看，self-attention能够替代crf的效果，同时由于multi-head self-attention的并行特性，因此，速度上去了。经典序列标注的模型结构是bilstm+crf，这篇说，bilstm+lan也阔以。近一段的工作证明，bert多数情况下都不需要crf，这篇文章也算是提供了一个佐证，既然crf都可以被self-attention替代，那直接把encoder也用self-attention替代完事儿了，结构还显得更加清晰。再一次证明：《attention is all you need》，逃。


26.《Multi-instance Multi-label Learning for Relation Extraction》

2012年的工作，引用量400+。将distant-supervision得到的数据，建模为一个miml问题。主要技术：graphical model+EM。文章引用了周志华老师的miml的相关工作。

延伸思考：

（1）snorkel的相关技术可以结合distant-supervision，比如denoising。

（2）更加general的relation extraction可以关注fewrel的进展。

（3）distant-supervision需要关注multi-instance的相关工作。

相关工作：《Relation Extraction with Multi-instance Multi-label Convolutional Neural Networks》

25.《Distant supervision for relation extraction without labeled data》

2009年的工作，引用非常多，但是并非distant-supervision第一次提出，作者之一是Dan Jurafsky。文章在关系抽取的时候，还是基于人工构建的feature，包括syntactic和lexicial feature等。

相关启发：

（1）regular expr是learnable的；instance和pattern的snowball式玩法。

（2）distant-supervision下，multi-instance的利用是亮点。

24.《KG-BERT: BERT for Knowledge Graph Completion》

这篇工作采用了和之前事件判别模型类似的思路，不同之处在于直接基于三元组做，不包含上下文的具体描述，在相关数据集上取得了SOTA结果。虽然作者给出了一些解释，不过文章实验做的不够充足，并且解释似乎不是特别具有说服性。

结论：灌水。

23.**如何使用更多的外部数据提升模型效果？**

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

22.《ZEN: Pre-training Chinese Encoder Enhanced by N-gram Representations》

号称目前为止最强中文NLP预训练模型。整体思路上可以从deep&wide结构来理解。deep结构和传统的bert类似，wide结构用来encode n-gram的信息。其实，提供了缓解中文bert在词语粒度上modeling不足的问题。

延伸思考：对词语粒度的语义单元modeling是关于中文bert的老问题了。一般有两个思路：第一个思路，直接在deep结构的input端添加embedding信息，但是这样的问题在于可能会引入noise信息；第二个思路，使用wide结构，单独训练词语粒度的embedding，最后在output端进行信息的融合。这里选择了第二种，其实也是一种比较general的思路。

虽然不清楚在哪个粒度建模比较合适，但是从最近的一些工作来看，融合一些高层semantic信息（多粒度的信息）不是一件很坏的事情。不管怎样，整体上是我喜欢的思路，简单有效。

具体图如下：

![img_zen](https://wx4.sinaimg.cn/mw690/aba7d18bly1g8om2b1jsej20iv0gradb.jpg)

21.《ELECTRA: PRE-TRAINING TEXT ENCODERS AS DISCRIMINATORS RATHER THAN GENERATORS》,ICLR2020

主要内容：Thorough experiments demonstrate this new pre-training task is more efﬁ- cient than MLM because the model learns from all input tokens rather than just the small subset that was masked out.

延伸思考：和non-parallel style transfer的工作类似。不过这篇的主要目的是学到一个好的MLM。图示如下：

![img_electra](https://wx3.sinaimg.cn/mw690/aba7d18bly1g8jwz090qcj20
