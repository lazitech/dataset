自己构建的数据集已经传到本GitHub仓库：

dataset0: 通用数据集
dataset1: 双色字符识别数据集
dataset2: 混乱背景字符识别数据集
dataset3: Ishihara_MNIST数据集
dataset4: 散点图排序数据集

将整个数据集下载后使用 mixed_train_final.json 文件进行微调，其中包括了图片所在位置、提问、回答及一些附加信息。

---

chartQA数据集请使用：https://huggingface.co/datasets/HuggingFaceM4/ChartQA 

各人根据各人的分工在dataset1-4训练集上微调之后，在dataset1~4的测试集和chartQA测试集上评测。因为qwen2.5vl的回答肯定不会是一字不差的，所以我们需要调用另一个LLM来评判它的回答是否和标准答案是相符的，这里我们可以用Qwen2.5-7B-Instruct，从硅基流动调用API，参考仓库代码： judge.py，其中包含了API调用地址、模型、我的APIkey（当然你也可以用你自己的）、评判过程使用的提示词（严格按照这个提示词使用）。

最终结果填到：https://docs.qq.com/sheet/DR2RWSEJ4dkJtVHdM?tab=BB08J2

形成一个如下的表格：

| 数据集      | 原始模型 | LoRA微调 | 映射层微调 |
| -------- | ---- | ------ | ----- |
| dataset1 |      |        |       |
| dataset2 |      |        |       |
| dataset3 |      |        |       |
| dataset4 |      |        |       |
| chartQA  |      |        |       |

同时将qwen2.5vl的回答和Qwen/Qwen2.5-7B-Instruct的评判结果保存到csv文件，每个数据集分别保存，最后我来检查一遍。（相当于完成这一部分需要填写对应的表格，然后每人给我五份csv文件）

如遇到任何问题请在群里交流！

这一部分ddl为14号中午之前，各位辛苦了！！！
