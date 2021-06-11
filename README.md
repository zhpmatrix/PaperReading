285.病历相似性（基于电子病历数据）

目前看到的主流思路是：梳理出EMR的各个维度，然后按照维度计算每个维度的相似性，每个维度都有自己的相似度计算方式，之后按照加权的方式求解。

个人想法：

（1）纯文本的方式。计算tf-idf（ES based solution）

（2）计算表征。但是由于EMR文本较多，医学文本对于精确性要求比较高，因此需要hierarchical representation fusion的思想。（不管怎样，首先需要一个好的encoder）

《Measurement and application of patient similarity in personalized predictive modeling based on electronic medical records》

![截屏2021-06-1111 17 00](https://user-images.githubusercontent.com/4077026/121626512-25a7c300-caa8-11eb-936f-cc2ebc1f2f71.png)


284.《A Uniﬁed Generative Framework for Various NER Subtasks》，邱锡鹏老师组的工作

主要内容：用seq2seq(bart)解决三种常见ner的case（flat ner + nested ner + discontinuous ner）

想法：

（1）在之前的工作中，围绕这三种情况，有很多的paper。但是这篇文章采用seq2seq来解决，思路上之前也已经有相关工作了，但是这篇文章主要采用bart作为plm。毕竟seq2seq是万能的，哈哈。

（2）围绕bert做的中文nlp比较多，为啥？原因之一是因为bert有中文版，但是想用一下bart，就需要自己训练一个中文的bart了。每当这个时候，就不禁想到英文世界的话语权是怎么来的，到底意味着啥？

（3）技术创新个人认为谈不上：seq2seq(plm:bart)+ner(是一个体力活儿，不过还是要做很多工作的)
![截屏2021-06-0918 35 01](https://user-images.githubusercontent.com/4077026/121339849-7c4eb900-c951-11eb-8ace-3e7b6644cf5b.png)




283.《SMedBERT: A Knowledge-Enhanced Pre-trained Language Model with Structured Semantics for Medical Text Mining》,丁香园的预训练语言模型

[参考文章](https://mp.weixin.qq.com/s/F51Behy6pz1F9LgO3ESkjA)

知识增强预训练语言模型。研究了丁香园，联合阿里和东南大学做的工作，丁香园利用5G的医疗领域中文文本+内部的知识图谱，通过巧妙的模型设计，得到的模型能够显著提升NER/NRE等上游任务的指标。我们可以利用开源爬取的数据(目前量<5G)，同时结合OMAHA，做类似的工作以支持上游模型。

282.《Modeling Joint Entity and Relation Extraction with Table Representation》,EMNLP2014

人傻就要多读书，比如，在2014年的工作中，已经用table的方式解决joint问题了，如下：

![截屏2021-06-0716 16 39](https://user-images.githubusercontent.com/4077026/120983106-f6ddd400-c7ab-11eb-8021-eb6e2a9bf4cb.png)




四篇information extraction相关的工作：

281.《Read, Retrospect, Select: An MRC Framework to Short Text Entity Linking》

280.《Integrating Graph Contextualized Knowledge into Pre-trained Language Models》，小样本信息抽取相关的工作

279.《A novel cascade binary tagging framework for relational triple extraction》

278.《Entity-Relation Extraction as Multi-Turn Question Answering》

277.《Lifelong Learning based Disease Diagnosis on Clinical Notes》，腾讯天衍实验室的工作，TODO

276.《PRGC: Potential Relation and Global Correspondence Based Joint Relational Triple Extraction》，腾讯天衍实验室的工作，TODO

275.《MIE: A Medical Information Extractor towards Medical Dialogues》

annotate online medical consultation dialogues in a window-sliding style.

274.《A Survey of Data Augmentation Approaches for NLP》

比较新的NLP数据增强文章，按照任务类型划分增强的方式。

273.《HyperCore: Hyperbolic and Co-graph Representation for Automatic ICD Coding》， ACL2021

ICD编码映射，医疗NLP的特色任务。

ICD的特点：

（1）层次

（2）共现(因果)

主要方法：GCN的应用


272.《Few-Shot Named Entity Recognition: A Comprehensive Study》，Jiawei Han组的工作

NER中的少样本问题，三种解决方案：

（1）基于proto的few-shot learning方法（此前在研究文本分类的few-shot问题时，该方向上的工作也一直比较受欢迎）

（2）带噪音，有监督预训练

（3）伪标签，自训练


271.《Summarizing Medical Conversations via Identifying Important Utterances》，COLING2020

主要内容：从医疗问答中抽取摘要。

数据：从春雨医生爬取

方法：抽取式摘要

270.《Enquire One’s Parent and Child Before Decision_ Fully Exploit Hierarchical Structure for Self-Supervised Taxonomy Expansion》，腾讯，刘邦

**分类树扩展**的工作，用于腾讯的疫情问答场景。刘邦的博士论文可以一读。

269.Federated Learning

《Privacy-Preserving Technology to Help Millioins of People_Federated Prediction Model for Stroke Prevention》

FL使用传统模型，也是目前主要做的工作

《Empirical Studies of Institutional Federated Learning For Natural Language Processing》

FL使用TextCNN的经验性工作

《FedED: Federated Learning via Ensemble Distillation for Medical Relation Extraction》

内容：FL应用于医疗关系抽取

结果：实现了隐私保护，但是指标下降

核心：在通信，不在计算

科普：[《Introduction to FL》](https://www.youtube.com/watch?v=STxtRucv_zo)，本质上还是分布式学习的一种。

结论：**除非必要，否则目前在工业界推进的ROI应该不算高。不单纯是一个算法问题，还是一个架构问题。但是在医疗行业目前现状下（数据孤岛现象），仍有必要关注**

268.《MedDG: A Large-scale Medical Consultation Dataset for Building Medical Dialogue System》，**Xiaodan Liang**等

构建了一个中文医学对话数据集，特点是：标注了每个对话可能涉及的实体。

基于该数据集，定义了两个任务：

（1）next entity prediction。文章中用multi-label classification的方式实现

（2）doctor response generation。标准的文本生成类任务+融合任务（1）中的实体信息（最简单的方式：直接concat实体）

其他：ICLR2021要基于该数据集举办一个比赛，可以关注。

想法：其实是对生成领域强化对实体信息的利用。传统做生成的同学有一些对应的方式强化对实体信息的利用。不过，文章中的建模方式更偏intent识别。

《MedDialog: Large-scale Medical Dialogue Datasets》，EMNLP2020，这篇工作也是构建了一个中文医疗对话数据集，不过没有实体信息。

267.《BioBERT：a pre-trained biomedical language representation model for biomedical text mining》

预训练任务没有做任何改进，但是在下游的三个理解任务上均取得了提升，比较适合工业界操作的工作。

![img267](https://ftp.bmp.ovh/imgs/2021/03/20a8a5f451f89ef3.png)

补充：《Conceptualized Representation Learning for Chinese Biomedical Text Mining》，阿里巴巴，张宁豫

![img267+](https://ftp.bmp.ovh/imgs/2021/03/5d1227f4839fe5de.png)



266.《Building Watson：An Overview of the DeepQA Project》，2011年，IBM Watson的DeepQA项目，具体实现细节

讨论了架构和工程实现的问题，其中的特色在于对**证据**的重视。

265.《Strategies For Pre-training Graph Neural Networks》，ICLR2020

主要内容：预训练图的工作

motivation：node的pretrain和graph的pretrain都要；之前的一些工作只考虑node或者graph的单个类型的pretrain

训练任务：

（1）node：context graph的定义，学习context；attribute mask任务
（2）graph：supervised graph-level properties prediction + structural graph similarity prediction

基础模型： GIN

直观感受：中规中矩；目前还没看到预训练图的工作应用于电商领域等

264.《A User-Centered Concept Mining System for Query and Document Understanding at Tencent》，KDD2019

腾讯刘邦的工作，刘邦的博士论文也有share，主要做概念挖掘，偏向于工程系统的工作。[相关文章](https://mp.weixin.qq.com/s?__biz=MjM5ODkzMzMwMQ==&mid=2650419989&idx=2&sn=ffd0e6f04c81465ca263666a97736418&chksm=becdb54f89ba3c590023cd593456f2252ffa4d3b0eba5923449c251fb874e315ab25f5e24bd2&scene=0&xtrack=1&key=f1baf9836e24d95b25894f5ed75172f24088bf020a51854847f3b766f363dcada55a2bb32d0b7035a4e950a75d243fa430b8ad5afc8fa5980995ce0205499d08f3575c568de5fd4256087f072173f32af6ccdad480c4a5c1f32f66cdb7ee4b734c6cb8adb7444d941bc69bd6d3640ecb06673e5d2678be04dd7a85447a829c28&ascene=0&uin=MTg2NTIxNzUxNw%3D%3D&devicetype=iMac+MacBookAir7%2C2+OSX+OSX+10.15.2+build(19C57)&version=11020201&lang=zh_CN&exportkey=Ax5WXz28zKfYGuYgxDzoJso%3D&pass_ticket=RSqRY9vnIzdQMSWnSw37xTPvHc6zF8ityHYJXITPs%2FtXSrUn%2FMohvluSBUnn1Zxt&wx_header=0)

263.《Read, Retrospect, Select: An MRC Framework to Short Text Entity Linking》

和《AutoRegressive Entity Retrieval》，ICLR2021一块儿读。

262.《CRSLab：An Open-Source Toolkit for Building Conversational Recommender System》

CRS系统的设定：

![img_262](https://ftp.bmp.ovh/imgs/2021/01/d3be7c25a4f1419c.png)


261.《Open Domain Event Extraction Using Neural Latent Variable Models》

开放域的事件抽取。（个人对隐变量模型不是很了解）

260.《Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba》，KDD2018

![260_1_img](https://ftp.bmp.ovh/imgs/2021/01/db47ef98a5f065f4.png)

![260_2_img](https://ftp.bmp.ovh/imgs/2021/01/fbc61749b4d401cf.png)

259.《AutoRegressive Entity Retrieval》，ICLR2021

用生成的方式（seq2seq）做el，entity disambiguation， page retrieval任务。将传统分类任务转化为一个生成任务是问题解决范式的转变，很有意思的工作。在自己的博客，[MRC is all you need?](https://zhpmatrix.github.io/2020/05/07/mrc-is-all-you-need/)中讨论了将很多经典NLP任务用MRC的方式来做。

258.《Metapath-guided Heterogeneous Graph Neural Network for Intent Recommendation》，KDD2019

基于user-item-query构建的图，三种类型的边：search,click, guide，用于淘宝意图检测。

257.《Heterogeneous Graph Attention Network》，WWW2019

异构图+gat，文章写作思路很赞。目前，我们正尝试将该工作用于销量预测与归因分析。

256.《POG：Personalized Outfit Generation for Fashion Recommendation ai Alibaba iFashion》，KDD2019

阿里dida平台建设，compatibility+personality都要考察。

相关PR稿(dida也是从luban演化而来)：https://hackernoon.com/finding-the-perfect-outfit-with-alibabas-dida-ai-assistant-71ba7c9e8cfa

255.《Spam Review Detection with Graph Convolutional Networks》,CIKM2018 Best Paper

重新翻开这篇文章，方法如下：

![img_255](https://ftp.bmp.ovh/imgs/2020/12/42595b8c0cace4ad.png)

文章要解决的问题是垃圾评论检测，构建了两个图。第一个图：用户-评论-商品图，是异构图；第二个图：评论-评论图，是同构图。分别用异构GCN和GCN学到各自的表征，做节点分类工作。

整体上，文章的思路和这篇《Abusive Language Detection with Graph Convolutional Networks》非常相似，但是二者都没有互相引用。这篇文章做的是Tweet分类（三分类），分别构建两个图。第一个图：用户-用户的同构图；第二个图：用户-Tweet的异构图。针对同构图，用node2vec去学到表征（node2vec不是仅仅适用于同构图，不过效果需要考察）；针对异构图，用gcn去学到表征。表征组合（embedding+n-gram）+分类器做节点分类。

对比二者，整体上的技术思路相似，不过显然后者在图构建上更加的自然。

254.《Graph Neural Networks：Taxonomy, Advances and Trends》，最新的GNN相关的综述文章

253.《Understanding Image Retrieval Re-Ranking：A Graph Neural Network Persperctive》

有意思的工作，作者提到：

  （1）Re-ranking can be reformulated as a high-parallelism Graph Neural Network (GNN) function.

  （2）On the Market-1501 dataset, we accelerate the re-ranking processing from 89.2s to 9.4ms with one K40m GPU.

252.《Why Are Deep Learning Models Not Consistently Winning Recommender Systems Competitions Yet?》，RecSys2020

非常棒的文章，多年以前自己就很好奇了。

251.《Enriching Pre-trained Language Model with Entity Information for Relation Classiﬁcation》

![img_251](https://ftp.bmp.ovh/imgs/2020/11/fd20de4ee9f75cfc.png)

250.《Diverse, Controllable, and Keyphrase-Aware:
A Corpus and Method for News Multi-Headline Generation》

新闻标题生成

249.《CharBERT: Character-aware Pre-trained Language Model》

248.《
