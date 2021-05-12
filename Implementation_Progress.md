| Operation | Implemented | Function | Gradient Implemented |
| --------- | :---------: | :------: | :------------------: |
|Abs | :heavy_check_mark: | `absolute` | :x: |
|Acos | :heavy_check_mark: | `acos` | :x: |
|Acosh | :heavy_check_mark: | `acosh` | :x: |
|Add | :heavy_check_mark: | `add` | :x: |
|And | :heavy_check_mark: | `logical_and` | :x: |
|ArgMax | :heavy_check_mark: | `argmax` | :x: |
|ArgMin | :heavy_check_mark: | `argmin` | :x: | 
|Asin | :heavy_check_mark: | `asin` | :x: |
|Asinh | :heavy_check_mark: | `asinh` | :x: |
|Atan | :heavy_check_mark: | `atan` | :x: |
|Atanh | :heavy_check_mark: | `atanh` | :x: |
|AveragePool | :heavy_check_mark: | `nn.average_pool` | :x: |
|BatchNormalization | :heavy_check_mark: (eval mode only) | `nn.batch_normalization` | :x: |
|BitShift | :heavy_check_mark: | `bitshift`, `right_shift`, `left_shift` | :x: |
|Cast | :heavy_check_mark: | `cast` | :x: |
|Ceil | :heavy_check_mark: | `ceil` | :x: |
|Clip | :heavy_check_mark: | `clip` | :x: |
|Compress | :heavy_check_mark: | `compress` | :x: |
|Concat | :heavy_check_mark: | `concat` | :x: |
|ConcatFromSequence | :x: | | :x: |
|Constant | :heavy_check_mark: | `constant` | :x: |
|ConstantOfShape | :heavy_check_mark: | `constant_of_shape` | :x: |
|Conv | :heavy_check_mark: | `nn.conv` | :x: |
|ConvInteger | :heavy_check_mark: (tests not working) | `nn.conv_integer` | :x: |
|ConvTranspose | :heavy_check_mark: | `nn.conv_transpose` | :x: |
|Cos | :heavy_check_mark: | `cos` | :x: |
|Cosh | :heavy_check_mark: | `cosh` | :x: |
|CumSum | :heavy_check_mark: | `cumsum` | :x: |
|DepthToSpace | :heavy_check_mark: | `nn.depth_to_space` | :x: |
|DequantizeLinear | :heavy_check_mark: | `nn.dequantize_linear` | :x: |
|Det | :heavy_check_mark: | `det` | :x: |
|Div | :heavy_check_mark: | `divide` | :x: |
|Dropout | :heavy_check_mark: | `nn.dropout` | :x: |
|Einsum | :heavy_check_mark: | `einsum` | :x: |
|Elu | :heavy_check_mark: | `nn.elu` | :x: |
|Equal | :heavy_check_mark: | `equal` | :x: |
|Erf | :heavy_check_mark: | `erf` | :x: |
|Exp | :heavy_check_mark: | `exp` | :x: |
|Expand | :heavy_check_mark: | `expand` | :x: |
|EyeLike | :heavy_check_mark: | `eye_like` | :x: |
|Flatten | :heavy_check_mark: | `flatten` | :x: |
|Floor | :heavy_check_mark: | `floor` | :x: |
|GRU | :heavy_check_mark: | `nn.gru` | :x: |
|Gather | :heavy_check_mark: | `gather` | :x: |
|GatherElements | :heavy_check_mark: | `gather_elements` | :x: |
|GatherND | :x: | | :x: |
|Gemm | :heavy_check_mark: | `gemm` | :x: |
|GlobalAveragePool | :heavy_check_mark: | `nn.global_average_pool` | :x: |
|GlobalLpPool | :heavy_check_mark: | `nn.global_lp_pool` | :x: |
|GlobalMaxPool | :heavy_check_mark: | `nn.global_max_pool` | :x: |
|Greater | :heavy_check_mark: | `greater` | :x: |
|GreaterOrEqual | :heavy_check_mark: | `greater_equal` | :x: |
|HardSigmoid | :heavy_check_mark: | `nn.hard_sigmoid` | :x: |
|Hardmax | :heavy_check_mark: | `nn.hardmax` | :x: |
|Identity | :heavy_check_mark: | `identity` | :x: |
|If | :x: | | :x: |
|InstanceNormalization | :heavy_check_mark: | `instance_normalization` | :x: |
|IsInf | :heavy_check_mark: | `isinf` , `isneginf` , `isposinf` | :x: |
|IsNaN | :heavy_check_mark: | `isnan` | :x: |
|LRN | :heavy_check_mark: | `lrn` | :x: |
|LSTM | :heavy_check_mark: | `nn.lstm` | :x: |
|LeakyRelu | :heavy_check_mark: | `nn.leakyrelu` | :x: |
|Less | :heavy_check_mark: | `less` | :x: |
|LessOrEqual | :heavy_check_mark: | `less_equal` | :x: |
|Log | :heavy_check_mark: | `log` | :x: |
|LogSoftmax | :heavy_check_mark: | `nn.logsoftmax` | :x: |
|Loop | :x: | | :x: |
|LpNormalization | :heavy_check_mark: | `lp_normalization` | :x: |
|LpPool | :x: | | :x: |
|MatMul | :heavy_check_mark: | `matmul` | :x: |
|MatMulInteger | :heavy_check_mark: | `matmul_integer` | :x: |
|Max | :heavy_check_mark: | `max` | :x: |
|MaxPool | :heavy_check_mark: | `nn.maxpool` | :x: |
|MaxRoiPool | :x: | | :x: |
|MaxUnpool | :x: | `nn.maxunpool` | :x: |
|Mean | :heavy_check_mark: | `mean` | :x: |
|Min | :heavy_check_mark: | `minimum` | :x: |
|Mod | :heavy_check_mark: | `mod` | :x: |
|Mul | :heavy_check_mark: | `multiply` | :x: |
|Multinomial | :heavy_check_mark: | `random.multinomial` | :x: |
|Neg | :heavy_check_mark: | `negative` | :x: |
|NegativeLogLikelihoodLoss | :heavy_check_mark: (unstable) | `nn.negative_loglikelihood_loss` | :x: |
|NonMaxSuppression | :x: | | :x: |
|NonZero | :heavy_check_mark: | `nonzero` | :x: |
|Not | :heavy_check_mark: | `not_` | :x: |
|OneHot | :heavy_check_mark: | `one_hot` | :x: |
|Or | :heavy_check_mark: | `logical_or` | :x: |
|PRelu | :heavy_check_mark: | `nn.prelu` | :x: |
|Pad | :x: | | :x: |
|Pow | :heavy_check_mark: | `power` | :x: |
|QLinearConv | :x: | | :x: |
|QLinearMatMul | :x: | | :x: |
|QuantizeLinear | :x: | | :x: |
|RNN | :x: | | :x: |
|RandomNormal | :heavy_check_mark: | `random.normal` | :x: |
|RandomNormalLike | :heavy_check_mark: | `random.normal_like` | :x: |
|RandomUniform | :heavy_check_mark: | `random.uniform` | :x: |
|RandomUniformLike | :heavy_check_mark: | `random.uniform_like` | :x: |
|Reciprocal | :heavy_check_mark: | `reciprocal` | :x: |
|ReduceL1 | :heavy_check_mark: | `l1_norm` | :x: |
|ReduceL2 | :heavy_check_mark: | `l2_norm` | :x: |
|ReduceLogSum | :heavy_check_mark: | `log_sum` | :x: |
|ReduceLogSumExp | :heavy_check_mark: | `log_sum_exp` | :x: |
|ReduceMax | :heavy_check_mark: | `max` | :x: |
|ReduceMean | :x: |  | :x: |
|ReduceMin | :heavy_check_mark: | `min` | :x: |
|ReduceProd | :heavy_check_mark: | `prod` | :x: |
|ReduceSum | :heavy_check_mark: | `sum` | :x: |
|ReduceSumSquare | :heavy_check_mark: | `sum_square` | :x: |
|Relu | :heavy_check_mark: | `relu` | :x: |
|Reshape | :heavy_check_mark: | `reshape` | :x: |
|Resize | :x: | | :x: |
|ReverseSequence | :x: | | :x: |
|RoiAlign | :x: | | :x: |
|Round | :heavy_check_mark: | `round` | :x: |
|Scan | :x: | | :x: |
|Scatter | Deprecated | | :x: |
|ScatterElements | :heavy_check_mark: | `nn.scatter` | :x: |
|ScatterND | :heavy_check_mark: | `nn.scatter_nd` | :x: |
|Selu | :heavy_check_mark: | `nn.selu` | :x: |
|SequenceAt | :x: | | :x: |
|SequenceConstruct | :x: | | :x: |
|SequenceEmpty | :x: | | :x: |
|SequenceErase | :x: | | :x: |
|SequenceInsert | :x: | | :x: |
|SequenceLength | :x: | | :x: |
|Shape | :heavy_check_mark: | `shape` | :x: |
|Shrink | :x: | | :x: |
|Sigmoid | :heavy_check_mark: | `nn.sigmoid` | :x: |
|Sign | :heavy_check_mark: | `sign` | :x: |
|Sin | :heavy_check_mark: | `sin` | :x: |
|Sinh | :heavy_check_mark: | `sinh` | :x: |
|Size | :heavy_check_mark: | `size` | :x: |
|Slice | :heavy_check_mark: | `slice` and `Array.__getitem__` | :x: |
|Softplus | :heavy_check_mark: | `nn.softplus` | :x: |
|Softsign | :heavy_check_mark: | `nn.softsign` | :x: |
|SpaceToDepth | :x: | | :x: |
|Split | :x: | | :x: |
|SplitToSequence | :x:| | :x: |
|Sqrt | :heavy_check_mark: | `sqrt` | :x: |
|Squeeze | :heavy_check_mark: | `squeeze` | :x: |
|StringNormalizer | :x: | | :x: |
|Sub | :heavy_check_mark: | `subtract` | :x: |
|Sum | :x: |  | :x: |
|Tan | :heavy_check_mark: | `tan` | :x: |
|Tanh | :heavy_check_mark: | `tanh` | :x: |
|TfIdfVectorizer | :x: | | :x: |
|ThresholdedRelu | :heavy_check_mark: | `nn.thresholded_relu` | :x: |
|Tile | :heavy_check_mark: | `tile` | :x: |
|TopK | :x: | | :x: |
|Transpose | :heavy_check_mark: | `transpose` | :x: |
|Trilu | :x: | | :x: |
|Unique | :heavy_check_mark: | `unique` | :x: |
|Unsqueeze | :heavy_check_mark: | `unsqueeze` | :x: |
|Upsample | Deprecated | | :x: |
|Where | :heavy_check_mark: | `where` | :x: |
|Xor | :heavy_check_mark: | `logical_xor` | :x: |
