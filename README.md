# 技术路线

随着预训练模型如BERT（Devlin et al., 2019）, GPT（Radford et al., 2018）以及T5（Raffel et al., 2019）在NLP任务中的成功应用，其在信息抽取任务上的潜力也引起了学术界的关注。预训练模型通过在大量文本数据上进行预训练，能够有效地捕捉语义知识，并在此基础上进行微调，以适应特定任务。例如，BertIE（Ma et al., 2020）和OpenIE6（Sakor et al., 2020）等方法使用BERT模型，通过设计特定的目标函数和训练策略，显著提升了信息抽取任务的性能。

然而，尽管预训练模型在多种NLP任务中都取得了显著的效果，但它们在生成式信息抽取任务中的表现仍存在一些局限性。比如，BERT等模型的双向自注意力机制限制了其在生成任务中的应用。

然而，尽管这些模型在生成式信息抽取任务上取得了一定的成果，但还存在一些挑战。例如，这些模型通常需要大量的训练数据，计算复杂性高，并且可能受到生成模型的限制。为了解决这些问题，一些研究者开始尝试使用预训练模型，如GPT（Radford et al., 2018），BERT（Devlin et al., 2019），以及我们在本文中将要探讨的UniLM模型。这些模型通过在大量的未标注文本数据上进行预训练，能够学习到丰富的语义知识，从而在各种生成式信息抽取任务上取得了显著的性能提升。

## 自注意力机制

自注意力机制是Transformer模型的核心部分，其主要目标是对输入序列中的每个元素，计算其与其他元素的相互关联程度。这种机制能够有效捕捉到长距离的依赖关系，并且与传统的循环神经网络相比，计算效率更高。

在具体实现中，自注意力机制主要通过以下步骤来计算输出序列。首先，对于输入序列X = {x1, x2, ..., xn}，其中xi表示序列中的第i个单元（例如单词或者字），自注意力机制将每个单元映射到一个query向量q_i、一个key向量k_i和一个value向量v_i：

q_i = W_q * x_i, k_i = W_k * x_i, v_i = W_v * x_i,

其中W_q、W_k和W_v是待学习的参数矩阵，*表示线性变换。这三个向量分别代表了模型对输入序列的不同视角，其中query向量主要用于计算权重，key向量用于计算相似度，value向量用于计算输出。

然后，自注意力机制计算每个query向量与所有key向量之间的点积，得到一个相似度矩阵，然后通过softmax函数将相似度矩阵归一化为权重矩阵：

a_i = softmax((q_i * K^T) / sqrt(d_k)),

其中d_k是key向量的维度，*表示矩阵乘法，K^T表示key向量的转置，sqrt表示平方根函数，softmax是一个将实数映射到(0,1)区间并且保证所有输出之和为1的函数，它使得权重矩阵中的每个元素代表了相应的key向量对于输出的贡献程度。

接着，自注意力机制根据权重矩阵对所有的value向量进行加权求和，得到输出向量：

o_i = sum(a_ij * v_j),

其中sum表示求和，*表示乘法。这一步的主要目标是根据每个key向量的贡献程度，对所有的value向量进行加权，得到最终的输出向量。这样，输出向量就可以捕捉到输入序列中的长距离依赖关系，并且由于softmax函数的归一化效果，所有输出向量的和为1，这使得模型可以更好地平衡各个位置的贡献。

## 自注意力掩码

为了实现双向和单向注意力机制的统一，LLM模型引入了一种新的自注意力掩码策略。这种掩码策略在计算自注意力的过程中，通过一个掩码矩阵M来控制每个query向量可以访问的key向量的范围。对于单向注意力机制，掩码矩阵M是一个上三角矩阵，使得每个query向量只能访问其之前的key向量，从而实现了单向的信息流。而对于双向注意力机制，掩码矩阵M是一个全1矩阵，使得每个query向量可以访问所有的key向量，从而实现了双向的信息流。

在训练过程中，LLM模型首先在大量的未标注文本数据上进行预训练。预训练的主要目标是学习语言模型，即学习预测给定前文的下一单词。预训练的过程主要通过最大化以下对数似然度来学习模型参数：

max θ sum(log p(xi | x_{<i}, θ)),

其中x_{<i}表示位置i之前的文本，θ表示模型参数，p(xi | x_{<i}, θ)表示模型在给定位置i之前的文本和参数θ的情况下，预测位置i的单词的概率。这种训练方式使得模型能够在大量的文本数据上学习到语言的一般规律和模式。

预训练完成后，LLM模型会在特定任务的标注数据上进行微调。微调的主要目标是学习特定任务的知识，即学习预测特定任务的标签。微调的过程主要通过最小化以下损失函数来优化模型参数：

min θ L(y, f(x, θ)),

其中x表示输入，y表示标签，θ表示模型参数，f(x, θ)表示模型在给定输入和参数的情况下，预测的标签，L(y, f(x, θ))表示标签和预测的标签之间的损失函数。这种训练方式使得模型能够在标注数据上学习到特定任务的知识。

## Flash Attention和ROPE

在深度学习领域，Transformer 已经成为了一种重要的模型结构，广泛应用于各种任务中，包括自然语言处理 (NLP)、计算机视觉 (CV) 和多模态问题。然而，Transformer 的计算复杂度随着输入序列长度的增加而呈平方增长，这使得其在处理长序列时面临巨大的挑战。为了解决这个问题，我们引入了 FlashAttention，一种新型的优化技术，通过优化存储器使用，提高了 Transformer 的执行效率。

另一方面，如何更好地表示位置信息，是 NLP 中的一个重要问题。为了解决这个问题，我们引入了 Rotary Position Embedding (RoPe)，一种新型的位置编码方案。RoPe 的主要思想是：用绝对位置编码来表征相对位置编码,主要就是对attention中的q, k向量注入了绝对位置信息，然后用更新的q,k向量做attention中的内积就会引入相对位置信息了。

FlashAttention是一种新型的注意力算法，旨在减少对高带宽内存（High Bandwidth Memory，HBM）的读写操作，从而提高Transformer模型的效率。该算法的主要思想是使注意力机制具备IO意识，考虑到GPU内存级别之间的读写操作。通过减少对HBM的读写操作，FlashAttention可以有效降低模型的内存访问瓶颈，提升训练速度和推理效率。另一项关键技术RoPE是一种新型的位置编码方法，通过旋转操作将位置信息融入模型，进一步提升模型在序列建模任务上的性能。RoPE通过将位置向量进行旋转操作，将位置信息与输入元素的表示进行融合，使模型能够更好地捕捉序列中的位置关系和上下文信息。

## FlashAttention

FlashAttention 的主要思路是尽量使用高速的静态随机存取存储器(SRAM)，减少对存取速度慢的 HBM（High Band Memory）的使用。具体来说，FlashAttention 的计算过程如下：

计算 S = QK^T ∈ R^(N×N)

计算内积 P = softmax(S) ∈ R^(N×N)

计算归一化的注意力 O = PV ∈ R^(N×d)

得到输出 Output

其中，Q, K, V ∈ R^(N×d)，N 是序列长度，d 是 head 的维度。FlashAttention 的关键在于尽量对矩阵乘法进行分块，每个小块都在 on-chip 的（192K）SRAM 进行。这主要通过两个相关的技术实现：Tiling 和 Recomputation。

## Rotary Position Embedding (RoPe)

RoPe 的工作原理主要基于复数运算。首先，将 Q 矩阵的第 m 个字的表示向量 qm 和 K 矩阵的第 n 个字的表示向量 kn 看作是复数。然后，对 qm 和 kn 进行变换：

f(q, m) = qm * e^(jmθ)

f(k, n) = kn * e^(jnθ)

其中，e^(jmθ) 和 e^(jnθ) 是位置 m 和 n 的绝对位置编码信息。然后，将变换后的数据做内积：

<f(q, m), f(k, n)> = f(q, m) * f(k, n)^*

其中，<> 表示内积计算，^* 表示复数的共轭。通过这样的计算，我们可以得到 (m - n) 的相对位置信息，而初始的变换也只是绝对位置信息的编码。因此，RoPe 实现了用绝对位置编码来表征相对位置编码。

## Multi-Query Attention (MQA)

MQA 是一种新型的注意力机制，其主要思想是在注意力机制中引入多查询。具体来说，MQA 的计算过程如下：

计算 S = QK^T ∈ R^(N×N×M)

计算内积 P = softmax(S) ∈ R^(N×N×M)

计算归一化的注意力 O = PV ∈ R^(N×d×M)

得到输出 Output

其中，Q, K, V ∈ R^(N×d)，N 是序列长度，d 是 head 的维度，M 是查询的数量。MQA 的关键在于引入多查询，每个查询都可以获取不同的注意力分布，从而提高模型的性能。

## 应用

在标准的Transformer模型（包括UniLM）中，attention机制是通过以下公式计算的：

计算Query，Key，Value：Q = W_q * X，K = W_k * X，V = W_v * X

计算attention score：S = softmax(Q * K^T / sqrt(d_k))

计算output：O = S * V

其中，X是输入，W_q，W_k，W_v是权重矩阵，d_k是Key的维度。

FlashAttention的主要思想是通过分块计算和重计算来减少对高带宽内存（HBM）的读写次数和降低内存使用。在FlashAttention中，softmax计算被分解为分块计算，每个块的元素最大值和分子、分母的计算，然后再合并。具体来说，假设输入矩阵X被分为两块X^(1)和X^(2)，那么分块softmax的计算过程如下：

计算每个块的元素最大值：m(X) = max([m(X^(1)), m(X^(2))])

计算分块softmax的分子：f(X) = [e^(m(X^(1)) - m(X)) * f(X^(1)), e^(m(X^(2)) - m(X)) * f(X^(2))]

计算分块softmax的分母：l(X) = [e^(X^(1) - m(X)) * l(X^(1)), e^(X^(2) - m(X)) * l(X^(2))]

计算softmax：softmax(X) = f(X) / l(X)

Multi Query Attention是Multi-Head Attention的变体形式，其中，Key和Value只有一个头，Query是多头。通过共享Key得到attention score时，MQA具有以下优势：更加省内存，计算量更低。在实现上，MQA和MHA主要是在计算Key和Value的过程中有计算量的差异，由于训练阶段由于数据是并行的，这种差异整体不明显，而在推理阶段，在memory cache的基础上，MQA中每个step的V的计算量为 dk，而MHA中计算量则为 dkh。

在MQA中，attention score的计算过程如下：

计算Query，Key，Value：Q = W_q * X，K = W_k * M，V = W_v * M

计算attention score：logits = Q * K^T

计算weights：weights = softmax(logits + mask)

计算output：O = weights * V

其中，X和M是输入，W_q，W_k，W_v是权重矩阵，mask是掩码矩阵。

在标准的Transformer模型中，位置编码通常是通过将一个位置编码向量添加到输入序列的每个元素上来实现的。这种方法可以有效地为模型提供序列中元素的位置信息，但是它不能直接处理相对位置信息。

RoPE的提出是为了解决这个问题。具体来说，RoPE将位置信息编码为一个旋转矩阵，然后将这个旋转矩阵应用到输入序列的每个元素上。这样，模型就可以直接处理相对位置信息，而不仅仅是绝对位置信息。

RoPE的公式如下：

计算位置编码：PE = W * pos

计算旋转矩阵：R = exp(i * PE)

计算旋转后的输入：X' = X * R

其中，pos是位置向量，W是权重矩阵，i是虚数单位，X是输入，'*'表示矩阵乘法，'exp'表示指数函数。

在实际使用中，我们发现不仅要求LLM能够识别单词和短语，更需要深入理解长文本序列，以有效捕获上下文中的复杂关系和语义。然而，传统的位置编码方法在处理不同长度序列时可能受到限制，导致模型难以准确捕捉长距离的依赖关系。为了克服这一挑战，我们引入了一项创新技术，即“自适应正弦余弦位置编码”，以提升模型在长序列建模方面的能力。

我们的方法采用了一种欺骗模型的思路，即根据输入序列的长度动态地调整位置编码的频率，从而更好地适应序列的变化，让模型认为其仍然在有效的长度内。

具体来说，我们通过计算自适应基数（base）来实现频率的调整，即

alpha=base /(1024−L)

base=base ×alpha^(d/(d−2))

其中，L 代表输入序列的长度，d 表示模型的维度，base 是初始基数。然后，我们根据自适应基数计算位置编码的频率，

freq=1/(base ^(i/d))

这样一来，每个位置的编码频率都会因序列长度的变化而不同，从而使模型能够更好地适应不同长度的序列。

在实际应用中，当输入序列的长度超过之前计算的最大长度时，我们会重新计算位置编码的频率，并生成新的位置编码。然而，对于较短的序列，我们仍然使用之前计算好的位置编码，以保持计算效率。这种策略保证了模型在处理不同长度的序列时能够充分应用自适应位置编码的优势。

综上所述，自适应正弦余弦位置编码技术为模型在处理长文本序列时的上下文建模能力提供了一种创新方法。通过根据序列长度调整位置编码的频率，我们能够更准确地捕捉不同位置之间的语义关系，从而增强模型对长距离依赖关系的感知能力。这项技术的引入不仅在各种自然语言处理任务中展现出更强的性能，还为模型在理解和处理长文本时开辟了全新的可能性。




# 量化和微调说明

## LORA

LoRA 是一种微调技术，旨在在预训练模型的基础上添加低秩结构，以实现高效的参数化微调。与传统的微调不同，LoRA 不需要修改原始预训练模型的参数，而是添加一个低秩适应层，这使得微调更为参数高效。

在本项目中：

模型结构中的 LoRA: 在这段代码中，我们可以看到检查模型是否具有 pretrained_model 属性。如果模型具有这个属性（并且有一个名为 v_head 的属性），这意味着它使用了 LoRA 或某种类似的技术，其中 pretrained_model 可能代表的是预训练的主模型，而其他属性（如 v_head）可能是该模型上的附加头部或部分。

    if hasattr(model, "pretrained_model"): # for models with valuehead (currently using LoRA only)
    
        backbone_model = getattr(model, "pretrained_model")

LoRA 微调类型的检查: 在另一段代码中，我们可以看到检查微调类型是否为 "lora"。根据微调类型，代码执行不同的保存或加载逻辑。

    if self.finetuning_args.finetuning_type == "lora":
        backbone_model.save_pretrained(output_dir, state_dict=get_state_dict(backbone_model))

和

    if self.finetuning_args.finetuning_type == "lora":
        backbone_model.load_adapter(self.state.best_model_checkpoint, getattr(backbone_model, "active_adapter"))

这些代码段提供了关于如何在此特定实现中使用 LoRA 的一些信息。具体来说，它们描述了如何在保存和加载模型时处理与 LoRA 相关的组件。

## 量化

量化 在深度学习中主要是指将模型的参数从浮点数转化为更小范围、更低精度的表示，例如从32位浮点数转为8/4位整数。量化的主要目的是减少模型的大小和推理时间，同时保持精度损失在可接受范围内。

量化过程中涉及两个主要步骤：

训练：在模型训练阶段，可以使用模拟量化，即在前向传播时使用低精度数值，但在反向传播和权重更新时仍使用原始的高精度数值。

推理：在模型推理阶段，模型使用量化后的低精度数值。

在本项目中：

量化模型的恢复: 当存在一个最新的检查点时，模型可能会恢复为 LoRA 训练或量化推理模式。

    if lastest_checkpoint is not None: # resume lora training or quantized inference
        model = PeftModel.from_pretrained(model, lastest_checkpoint, is_trainable=is_trainable)

检查点与量化的关系: 在这段代码中，我们可以看到关于量化模型和检查点之间的一些断言。具体来说，量化模型只接受一个检查点，而 LoRA 微调可以接受多个检查点。

    if finetuning_args.finetuning_type != "lora":
        assert len(model_args.checkpoint_dir) == 1, "Only LoRA tuning accepts multiple checkpoints."else:
        assert model_args.quantization_bit is None or len(model_args.checkpoint_dir) == 1, 
            "Quantized model only accepts a single checkpoint."

量化评估的警告: 如果模型在 4/8 位模式下进行评估，可能会导致得分较低，这里给出了一个警告。

    if model_args.quantization_bit is not None and (not training_args.do_train):
        logger.warning("Evaluating model in 4/8-bit mode may cause lower scores.")

这些代码段提供了关于在此特定实现中如何处理与量化相关的逻辑的一些信息。具体来说，它们描述了如何在恢复、保存和评估模型时处理量化。

量化位设置: 这部分代码定义了一个名为 quantization_bit 的字段，它允许用户指定用于量化模型的位数。

    quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "The number of bits to quantize the model."}
    )

量化类型设置: 除了量化位数外，还定义了一个名为 quantization_type 的字段，允许用户选择在 int4 训练中使用的量化数据类型，可选值为 "fp4" 和 "nf4"。

    quantization_type: Optional[Literal["fp4", "nf4"]] = field(
        default="nf4",
        metadata={"help": "Quantization data type to use in int4 training."}
    )

这些代码段为量化提供了配置选项，允许用户根据需要进行自定义设置。




# 文件目录信息

config.py：包含模型和训练的配置信息。

check.py：包含一些验证或检查逻辑。

other.py：包含一些其他的实用工具或函数。

peft_trainer.py： PEFT 的训练策略有关。

pairwise.py：与成对的数据操作或成对的损失函数有关。

common.py：包含一些公共的工具或函数。

template.py：包含一些调用大模型的模板代码或函数。

data_collator.py：与数据整理或预处理有关的代码。

ppo.py：与 PPO (Proximal Policy Optimization) 训练策略相关的代码。

seq2seq.py：与序列到序列模型相关的代码。

export_model.py：此文件包含导出模型的逻辑，以适应不同的部署场景。

web_demo2.py 和 web_demo.py：这些文件可能包含为模型提供界面的代码。

infer.sh：shell脚本，用于模型推理。

cli_wo_history.py：命令行界面的代码，但不记录历史。

train_pt.py、train_ppo.py、train_sft.py 和 train_rm.py：不同训练策略的代码。

test.py：包含模型测试的代码。

api_demo.py 和 api.py：为模型提供API接口。

## seq2seq.py
seq2seq.py 文件主要涉及序列到序列模型的实现和评估。以下是此文件的主要组件和功能的概述：

导入模块：文件开始部分导入了必要的库和工具。

日志设置：使用 get_logger 函数创建日志记录器。

ComputeMetrics 类：此类将分词器包装到度量函数中，主要用于 Seq2SeqPeftTrainer 中。

__call__ 方法：使用模型预测来计算度量标准。主要关注 Rouge 和 BLEU 分数的计算。

Seq2SeqPeftTrainer 类：继承自 PeftTrainer，用于序列到序列任务的训练。

compute_loss 方法：计算损失。

log_metrics 方法：记录模型的度量标准。

training_step 和 training_step_and_backward 方法：定义模型的训练步骤。

prediction_step 方法：定义模型的预测步骤，包括在生成的令牌中删除提示部分。

save_predictions 方法：将模型的预测结果保存到 output_dir。

## config.py

### DatasetAttr 类

该类定义了数据集的属性。

load_from：从哪里加载数据。

dataset_name：数据集的名称。

dataset_sha1：数据集的哈希值。

source_prefix：可能用于指定数据的来源前缀。

prompt_column, query_column, response_column, history_column：指定数据集中的各列名称。

### ModelArguments 类

此类定义了与模型相关的参数。

model_name_or_path：预训练模型的路径或来自 huggingface.co/models 的模型标识符。

cache_dir：存储从 huggingface.co 下载的预训练模型的位置。

use_fast_tokenizer：是否使用快速分词器（由 tokenizers 库支持）。

use_auth_token：是否使用运行 huggingface-cli login 时生成的令牌。

model_revision：要使用的特定模型版本（可以是分支名称、标签名称或提交id）。

quantization_bit：量化模型的位数。

quantization_type：用于 int4 训练的量化数据类型。

double_quantization：是否使用双重量化。

### GeneratingArguments 类

此类定义了与生成相关的参数。

do_sample：是否使用采样，否则使用贪婪解码。

temperature：用于调制下一个令牌的概率的值。

top_p：保持概率相加为 top_p 或更高的最小的最可能的令牌集。

top_k：用于 top-k 过滤的最高概率词汇令牌的数量。

num_beams：波束搜索的波束数。1 表示没有波束搜索。

max_length：生成令牌可以具有的最大长度。

max_new_tokens：要生成的令牌的最大数量，忽略提示中的令牌数量。

repetition_penalty：重复惩罚的参数。1.0 表示没有惩罚。

length_penalty：与基于波束的生成一起使用的长度的指数惩罚。

## common.py

模型加载：文件中的一部分关注于加载模型。根据不同的条件和参数，它决定从哪个检查点或路径加载模型。此外，它还处理了如何为不同的阶段准备模型，包括添加值头或加载奖励模型等。

适配器初始化：模型还经历了适配器的初始化过程，这可能涉及到模型的微调或其他特定任务的适应。

数据集预处理：根据训练的阶段，文件定义了如何预处理数据集。例如，对于预训练、有监督的微调、无监督的微调、对偶训练等，都有不同的预处理函数。

打印数据集示例：为了给用户提供更好的可见性，文件中有一些功能可以打印数据集的示例，以便用户可以查看经过预处理后的数据如何呈现。

其他实用函数：文件中还包含了其他一些实用函数，例如用于调整学习率的函数、用于计算奖励的函数等。

## peft_trainer.py 

LogCallback 类： 此类为训练器回调，其主要功能是在训练期间记录日志。这样做有助于在训练过程中动态地监控模型的性能和进度。

PeftTrainer 类： 继承自 Seq2SeqTrainer，这个类是为 PEFT 训练过程提供特定实现的训练器。以下是它的关键功能：

初始化：设置训练器的各种参数和配置。

training_step 和 training_step_and_backward：定义模型的训练步骤。

prediction_step：定义模型的预测步骤。

日志记录功能：例如 _log 和 log_metrics，用于记录训练日志和模型指标。

模型保存和加载：例如 save_model 和 _load_best_model，允许保存训练的模型和从检查点加载最佳模型。

此文件还包含了一些其他的实用函数和方法，例如处理适配器、Lora微调等。

## 使用ntk技术在训练时增加上下文的能力：

### defined __init__

    def adaptive_ntk_init(self, dim, max_position_embeddings=4096, base=10000, device=None):
        self.dim = dim
        self.base = base
        old_init(self, dim, max_position_embeddings, base, device)

这个新的初始化方法首先设置了 dim 和 base 属性。然后它调用了原始的 __init__ 方法（即 old_init）。

### defined forward

    def adaptive_ntk_forward(self, x, seq_len=None):
        ...

这是一个新的前向传播方法。以下是其详细的步骤分析：

检查 seq_len 是否大于 self.max_seq_len_cached。如果是，则：

根据 seq_len 创建一个形状为 [seq_len] 的张量 t。

使用给定的公式计算 alpha。

使用 alpha 和其他参数重新计算 inv_freq。

使用 t 和 inv_freq 生成嵌入 emb。

从 emb 中分离出余弦和正弦部分，并将它们存储为 cos_cached 和 sin_cached。

返回 cos_cached 和 sin_cached 的前 seq_len 部分。

如果 seq_len 不大于 self.max_seq_len_cached，则返回缓存中的 cos_cached 和 sin_cached 的前 seq_len 部分。

### 替换原始的 __init__ 和 forward

    transformers.models.llama.modeling_llama.LlamaRotaryEmbedding.forward = adaptive_ntk_forward
    transformers.models.llama.modeling_llama.LlamaRotaryEmbedding.__init__ = adaptive_ntk_init

这两行代码将 LlamaRotaryEmbedding 类的原始 __init__ 和 forward 方法替换为上面定义的新方法。
