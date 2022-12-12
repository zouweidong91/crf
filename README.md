# crf
* crf原理最通俗易懂介绍参考：  
* https://zhuanlan.zhihu.com/p/119254570?utm_medium=social&utm_oi=668925446389895168&utm_psn=1584475475874627584&utm_source=wechat_session  
* loss函数本质求解目标路径在所有路径中的概率值，优化方向使这个概率值最大  
* 代码实现参考tf或者bert4keras  stubs/tensorflow/contrib/crf/python/ops/crf.py  
* crf_log_norm(Computes the normalization for a CRF)归一化因子Z计算： 定义好cell单元后，再使用rnn.dynamic_rnn去迭代计算每一步的归一化值Z(i)，最终得到整个序列的Z  









