| Operation | Implementation Progress | Function |
| --------- | :---------------------: | :------: |
|Abs | :heavy_check_mark: | `absolute`
|Acos | :heavy_check_mark: | `acos`
|Acosh | :heavy_check_mark: | `acosh`
|Add | :heavy_check_mark: | `add`
|And | :heavy_check_mark: | `logical_and`
|ArgMax | :x: |
|ArgMin | :x: | 
|Asin | :heavy_check_mark: | `asin`
|Asinh | :heavy_check_mark: | `asinh`
|Atan | :heavy_check_mark: | `atan`
|Atanh | :heavy_check_mark: | `atanh`
|AveragePool | :x: | 
|BatchNormalization | :x: |
|BitShift | :x: |
|Cast | :heavy_check_mark: | `cast`
|Ceil | :heavy_check_mark: | `ceil`
|Clip | :heavy_check_mark: | `clip`
|Compress | :x: |
|Concat | :heavy_check_mark: | `concat`
|ConcatFromSequence | :x: |
|Constant | :heavy_check_mark: | `constant`
|ConstantOfShape | :heavy_check_mark: | `constant_of_shape`
|Conv | :x: |
|ConvInteger | :x: |
|ConvTranspose | :x: |
|Cos | :heavy_check_mark: | `cos`
|Cosh | :heavy_check_mark: | `cosh`
|CumSum | :heavy_check_mark: | `cumsum`
|DepthToSpace | :x: |
|DequantizeLinear | :x: |
|Det | :heavy_check_mark: | `det`
|Div | :heavy_check_mark: | `divide`
|Dropout | :x: |
|Einsum | :heavy_check_mark: | `einsum`
|Elu | :heavy_check_mark: | `nn.elu`
|Equal | :heavy_check_mark: | `equal`
|Erf | :heavy_check_mark: | `erf`
|Exp | :heavy_check_mark: | `exp`
|Expand | :heavy_check_mark: | `expand`
|EyeLike | :heavy_check_mark: | `eye_like`
|Flatten | :heavy_check_mark: | `flatten`
|Floor | :heavy_check_mark: | `floor`
|GRU | :x: |
|Gather | :heavy_check_mark: | `gather`
|GatherElements | :heavy_check_mark: | `gather_elements`
|GatherND | :x: |
|Gemm | :heavy_check_mark: | `gemm`
|GlobalAveragePool | :heavy_check_mark: | `nn.global_average_pool`
|GlobalLpPool | :heavy_check_mark: | `nn.global_lp_pool`
|GlobalMaxPool | :heavy_check_mark: | `nn.global_max_pool`
|Greater | :heavy_check_mark: | `greater`
|GreaterOrEqual | :heavy_check_mark: | `greater_equal`
|HardSigmoid | :heavy_check_mark: | `nn.hard_sigmoid`
|Hardmax | :heavy_check_mark: | `nn.hardmax`
|Identity | :heavy_check_mark: | `identity`
|If | :x: |
|InstanceNormalization | :heavy_check_mark: | `instance_normalization`
|IsInf | :heavy_check_mark: | `isinf`, `isneginf`, `isposinf`
|IsNaN | :heavy_check_mark: | `isnan`
|LRN | :heavy_check_mark: | `lrn`
|LSTM | :x: | 
|LeakyRelu | :heavy_check_mark: | `nn.leakyrelu`
|Less | :heavy_check_mark: | `less`
|LessOrEqual | :heavy_check_mark: | `less_equal`
|Log | :heavy_check_mark: | `log`
|LogSoftmax | :heavy_check_mark: | `nn.logsoftmax`
|Loop | :x: |
|LpNormalization | :heavy_check_mark: | `lp_normalization`
|LpPool | :x: |
|MatMul | :heavy_check_mark: | `matmul`
|MatMulInteger | :heavy_check_mark: | `matmul_integer`
|Max | :heavy_check_mark: | `max`
|MaxPool | :x: |
|MaxRoiPool | :x: |
|MaxUnpool | :x: | `nn.maxunpool`
|Mean | :heavy_check_mark: | `mean`
|NegativeLogLikelihoodLoss | :heavy_check_mark: | `mean`
|Min | :heavy_check_mark: | `minimum`
|Mod | :heavy_check_mark: | `mod`
|Mul | :heavy_check_mark: | `multiply`
|Multinomial | :heavy_check_mark: | `random.multinomial`
|Neg | :heavy_check_mark: | `negative`
|NegativeLogLikelihoodLoss | :x: | 
|NonMaxSuppression | :x: |
|NonZero | :x: |
|Not | :heavy_check_mark: | `not_`
|OneHot | :x: |
|Or | :heavy_check_mark: | `logical_or`
|PRelu | :heavy_check_mark: | `nn.prelu`
|Pad | :x: |
|Pow | :heavy_check_mark: | `power`
|QLinearConv | :x: |
|QLinearMatMul | :x: |
|QuantizeLinear | :x: |
|RNN | :x: |
|RandomNormal | :heavy_check_mark: | `random.normal`
|RandomNormalLike | :heavy_check_mark: | `random.normal_like`
|RandomUniform | :heavy_check_mark: | `random.uniform`
|RandomUniformLike | :heavy_check_mark: | `random.uniform_like`
|Reciprocal | :heavy_check_mark: | `reciprocal`
|ReduceL1 | :heavy_check_mark: | `l1_norm`
|ReduceL2 | :heavy_check_mark: | `l2_norm`
|ReduceLogSum | :heavy_check_mark: | `log_sum`
|ReduceLogSumExp | :heavy_check_mark: | `log_sum_exp`
|ReduceMax | :heavy_check_mark: | `max`
|ReduceMean | :x: | 
|ReduceMin | :heavy_check_mark: | `min`
|ReduceProd | :heavy_check_mark: | `prod`
|ReduceSum | :heavy_check_mark: | `sum`
|ReduceSumSquare | :heavy_check_mark: | `sum_square`
|Relu | :heavy_check_mark: | `relu`
|Reshape | :heavy_check_mark: | `reshape`
|Resize | :x: |
|ReverseSequence | :x: |
|RoiAlign | :x: |
|Round | :heavy_check_mark: | `round`
|Scan | :x: |
|Scatter | Deprecated |
|ScatterElements | :heavy_check_mark: | `nn.scatter`
|ScatterND | :heavy_check_mark: | `nn.scatter_nd`
|Selu | :heavy_check_mark: | `nn.selu`
|SequenceAt | :x: |
|SequenceConstruct | :x: |
|SequenceEmpty | :x: |
|SequenceErase | :x: |
|SequenceInsert | :x: |
|SequenceLength | :x: |
|Shape | :heavy_check_mark: | `shape`
|Shrink | :x: |
|Sigmoid | :x: | 
|Sign | :x: |
|Sin | :heavy_check_mark: | `sin`
|Sinh | :heavy_check_mark: | `sinh`
|Size | :heavy_check_mark: | `size`
|Slice | :x: |
|Softplus | :heavy_check_mark: | `nn.softplus`
|Softsign | :heavy_check_mark: | `nn.softsign`
|SpaceToDepth | :x: |
|Split | :x: |
|SplitToSequence | :x:|
|Sqrt | :heavy_check_mark: |
|Squeeze | :x: |
|StringNormalizer | :x: |
|Sub | :heavy_check_mark: | `subtract`
|Sum | :x: | 
|Tan | :x: |
|Tanh | :x: |
|TfIdfVectorizer | :x: |
|ThresholdedRelu | :x: |
|Tile | :x: |
|TopK | :x: |
|Transpose | :x: |
|Trilu | :x: |
|Unique | :x: |
|Unsqueeze | :x: |
|Upsample | :x: |
|Where | :x: |
|Xor | :x: |