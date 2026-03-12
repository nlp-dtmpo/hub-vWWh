作业1：bert 文本分类和实体识别有什么关系，分别使用什么 loss？
1. 两个任务都共享同一个 BertModel
2. BERT 先把输入文本进行编码，然后分出两个任务：
第一个：预测整句 intent；
第二个：预测每个 token 的槽位标签。

都是用的 CrossEntropyLoss，
 self.criterion = nn.CrossEntropyLoss()

 seq_loss = self.criterion(seq_output, seq_label_ids)
 token_loss = self.criterion(active_logits, active_labels)

作业2：多任务训练 loss = seq_loss + token_loss 有什么坏处？如果存在训练不平衡的情况，如何处理？
loss简单相加的坏处：
1. 两个任务的损失量和数值范围可能不同，文本识别，一个样本只预测 1 个标签；实体识别，一个样本要预测很多个 token 标签，即使都用交叉熵，它们的数值规模和梯度强度也常常不同。
2. 实体识别容易主导训练结果。一个句子只有一个意图但是可能有多个token，即使token做了平均优化，依然可能导致在训练中压过意图识别
3. 多任务共享 BERT 参数；如果一个任务学习得快，另一个任务学习得慢，可能互相干扰
4. 如果某些 intent 样本很多，某些很少；或某些实体标签出现频率很低；直接相加会让高频类别、强势任务更占优势。
5. 有可能文本识别更准确；也有可能实体识别更准确；简单相加默认两个任务同等重要，可能导致最后结果不准确。
如果训练不平衡，如何处理？
1. 加权求和，给不同识别添加不同的权重。
2. 文本分类里给少数类更高权重；实体分类里给稀有标签更高权重。
