# 实现与数学定义

## 输入输出

输入为(I,Ta,Tb)，输出p=P(OOC|I,Ta,Tb)。文本两分支共享编码器、局部对齐和图文比较模块。默认CLIP冻结，只训练新增层；CLIP内部patch与词元由内部编码器接口读取，不直接把全局向量当局部特征。局部词元排除padding、BOS、EOS。

## RA-LFE

设X∈R^(B×N×d)，N=H×W，K为通道视角数，q=d/K。LayerNorm和1×1投影得到Z，每个位置分为K个q维视角S_n。

E_n = S_n + W_o[softmax((S_nW_Q)(S_nW_K)^T/√q)(S_nW_V)]

a = softmax_spatial(w_a^T Z)，c = Σ_n a_n Z_n。

D = GELU(DWConv3×3(Z))，ω_n = softmax_views(W_lD_n + W_cc)。

F_n = Concat_k(ω_nk E_nk)，g_n = sigmoid(w_g^TD_n+b_g)。

Y = X + γ W_out(F + g φ(c))。

γ初值0.01；残差保留原始表征通路。固定d、K时本模块的空间复杂度随N线性增长，但这不表示CLIP骨干或跨模态局部对齐也没有二次/乘积复杂度。RA-only保留上下文门控残差；LFE-only删除全局上下文项；两者关闭时返回X。

## 局部与全局差异

A = (VW_v)(TW_t)^T/√d。

视觉到文本：T_hat=softmax_tokens(A)TW_t；文本到视觉：V_hat=softmax_patches(A^T)VW_v。

局部差异为视觉位置平均|VW_v−T_hat|与有效文本位置平均|TW_t−V_hat|的拼接、映射。

定义S(a,b)=[a+b;|a−b|;a⊙b]。全局差异由图像与文本投影的S表示映射得到。拼接各自局部/全局差异得到z_a,z_b。

融合h=[S(z_a,z_b);S(t_a,t_b)]，p=sigmoid(MLP(h))。由S的交换不变性，两段描述交换不会改变推理输出。

## 损失

L = L_cls + λ_align L_align + λ_cons L_cons。

L_cls = Σ_i w_i BCEWithLogits(s_i,y_i) / max(Σ_i w_i,1)。y是合成标签，w是样本置信权重，不是人工OOC标签。

L_align采用双向、多正例InfoNCE：对每幅图像，所有相同图像ID的原始caption1为正例，分母为batch全部原始caption1；反向文本到图像同理。嵌入L2归一化，温度默认0.07。该关联监督仍可能受到新闻配图噪声影响。

L_cons = mean((p(I,Ta,Tb)−p(aug(I),Ta,Tb))²)。aug是归一化图像空间的小幅统一亮度平移，避免裁剪丢失证据；不改变文本。验证时不做随机增强，一致性损失为0，因此选择的是固定的分类+关联代理验证损失。

λ_align=0.1、λ_cons=0.05和学习率等是可修改的初始设置，未由真实COSMOS实验优化。

参考接口：https://huggingface.co/docs/transformers/model_doc/clip
数据协议：https://github.com/shivangi-aneja/COSMOS
