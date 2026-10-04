# LGAM-Net：COSMOS训练与测试

默认模型使用预训练CLIP ViT-B/16，新增模块需要训练。本项目不附COSMOS数据或预训练权重。

## 1. 文件组织

项目根目录只有两个文件：`train.py`、`test.py`。其他文件按功能放在以下文件夹。

```text
LGAM-Net/
  train.py
  test.py
  configs/   cosmos.json、requirements.txt
  models/    encoders.py、ra_lfe.py、lgam_net.py
  data/      records.py、weak_supervision.py、dataset.py、collate.py
  losses/    objective.py
  engine/    trainer.py、evaluator.py
  utils/     runtime.py、metrics.py
  tests/     test_pipeline.py
  docs/      README.md、METHOD.md、VALIDATION.md
```

训练自动创建`outputs/`目录。所有相对路径均相对于运行命令时的工作目录，建议始终在项目根目录运行。

## 2. 环境安装

建议Python 3.11，单张16 GB NVIDIA显卡。以下是用户指定环境的安装命令；需要能访问相应下载源。

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r configs/requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

不依赖Detectron2、torchvision或spaCy现场推理。默认使用数据中已经给出的entity_list。CUDA驱动需支持安装的PyTorch版本；显存不足时减小batch_size，提高accumulation。默认冻结CLIP；取消冻结会显著增加显存需求。

## 3. 数据准备

```text
D:/COSMOS/
  train_data.json
  val_data.json
  test_data.json
  train/1.jpg ...
  val/1.jpg ...
  test/0.jpg ...
```

支持JSON数组或每行一个对象的JSONL，UTF-8编码。JSON中的img_local_path必须相对于data.root，不能越出该目录。

训练/验证记录必须有img_local_path和articles；articles中每项使用原始caption及entity_list。测试记录使用caption1、caption2和context_label（或label），1=OOC，0=非OOC；也支持官方字符串`ooc/not-ooc`。caption_modified是实体匿名化文本，不能作为失配标签，本实现不使用该字段。模型采用patch区域，不使用maskrcnn_bboxes，不能在论文中声称使用了目标框分支。

数据集说明：https://github.com/shivangi-aneja/COSMOS

编辑`configs/cosmos.json`的root、train_json、val_json三个路径。文件名以实际下载数据为准。代码拒绝训练/验证路径交集；这只是路径级检查，正式实验还应按原始图像身份或内容哈希核查重复图片和转载造成的泄漏。

首次启动会下载`openai/clip-vit-base-patch16`。离线使用时把clip_name改为包含模型、配置、tokenizer及preprocessor文件的本地Hugging Face目录，并设置local_files_only=true。当前图像预处理固定224×224，与默认ViT-B/16匹配；更换其他输入分辨率的骨干需同步修改预处理。

## 4. 训练

```bash
python train.py --config configs/cosmos.json
```

输出`outputs/lgam_seed42/`：

- best.pt：验证集**弱监督代理损失**最低的完整checkpoint。
- last.pt：最近一个完整epoch的checkpoint，用于续训。
- history.jsonl：逐epoch训练和代理验证损失。
- weak_supervision_audit.json：前100幅训练图像的初始合成描述、伪标签、权重和来源，供人工检查。
- config.json、entity_pool.json：运行配置及仅由训练集构建的实体替换池。

```bash
python train.py --config configs/cosmos.json --resume outputs/lgam_seed42/last.pt
```

续训恢复模型、优化器、调度器、AMP及随机数状态，适用于相同配置下中断的运行。不要在续训中修改总epoch、batch_size或学习率；需要新实验时指定新的输出目录。checkpoint包含Python训练状态，只加载自己生成或可信来源的文件。

## 5. 测试

```bash
python test.py --checkpoint outputs/lgam_seed42/best.pt --test-json D:/COSMOS/test_data.json --data-root D:/COSMOS --output outputs/test_seed42 --batch-size 16
```

输出metrics.json与predictions.csv。指标范围为0–1，换算百分数需乘100。正类为OOC，混淆矩阵为`[[TN,FP],[FN,TP]]`。CSV逐条记录原图路径、真实标签、失配概率、预测标签和TP/TN/FP/FN。阈值默认0.5，测试前固定；不要利用测试标签选择阈值、epoch或合成规则。

输入是图片及两段描述的**整体上下文关系**，输出不直接指出哪段文字为假，也不是新闻事实核查结论。仅有caption1与caption2相似不能证明一致，二者不同也不能证明失配。

## 6. 弱监督的重要限制

COSMOS训练/验证集没有人工OOC标签。因此实现使用可追溯的**合成代理任务**：同图不同原始描述作为低权重伪非OOC；从一段原始描述中替换同类型实体作为伪OOC。实体池仅由训练集建立。若没有合法替换，分类权重设为0，该条仅参与图文关联学习。

同图描述可能天然互相矛盾，替换实体也不必然产生可由图像验证的错误；这些不是人工真值。默认弱标签权重是工程起点，需在开发阶段审查样本、检查噪声。合成任务也可能引入词汇捷径，不能只凭低代理验证损失推断真实OOC泛化。对同图弱负类不放心可设置same_image_weight=0，但此时需要额外设计可靠的负类监督，不能直接作为完善方案发表。

本实现保留caption1为原始关联文本，caption2在正类代理中扰动，因此对比损失只用caption1。同一batch内同图的重复样本作为多正例，避免互相成为负例。

## 7. 消融配置

保持数据、训练预算、种子、阈值和编码器一致，复制配置并修改输出目录：

| 变体 | use_ra | use_lfe | use_local_alignment |
|---|---|---|---|
| 无RA-LFE增强 | false | false | true |
| 仅RA增强 | true | false | true |
| 仅LFE增强 | false | true | true |
| 完整LGAM-Net | true | true | true |
| 无局部对齐 | true | true | false |

“无RA-LFE增强”是本网络内部消融，**不是COSMOS官方Baseline**。COSMOS官方方法需独立实现/运行，不能给某个开关变体直接改名作为官方基线。当前无Dropout保证确定性的对称融合；随机图像增强只用于一致性训练。

## 8. 验证及正式实验

```bash
python -m unittest tests.test_pipeline -v
```

测试使用自动生成图片和随机小模型，不下载CLIP权重，不使用真实测试结果。覆盖前向/反向、RA/LFE开关、描述交换对称性、JSON/JSONL、合成标签、checkpoint及预测文件、真实transformers CLIP接口。

正式研究需完整运行COSMOS并记录硬件、软件版本、种子、分割清单、checkpoint和预测。若报告100次平均，须实际完成100次独立训练/评价并同时报告标准差；一个checkpoint重复确定性测试100次不构成独立实验。91.03%不保证能由此原型取得。

## 文本长度

CLIP默认最多77个token（包含特殊符号），超长caption会截断；若被替换实体落在截断区外，合成监督可能变成噪声。正式训练前应通过审查文件核对长文本及替换位置，必要时设计保留关键实体的句子选择策略，并在论文中说明。
