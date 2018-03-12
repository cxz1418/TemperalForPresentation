import matplotlib
matplotlib.use('Agg')
import os
import sys
import numpy as np
import json
import matplotlib.pyplot as plt

import caffe
from caffe import layers as L
from caffe import params as P

from vqa_data_provider_layer import VQADataProvider
from visualize_tools import exec_validation, drawgraph
import config


# ---- Genome ----
# Genome represents a more balanced distribution
# of the 6W question types. Moreover, the average
# question and answer lengths for Visual Genome
# are larger than the VQA dataset

def qlstm(mode, batchsize, T, question_vocab_size):

    #prototxt 없이 network 생성시 사용
    n = caffe.NetSpec()
    mode_str = json.dumps({'mode':mode, 'batchsize':batchsize})


    #지정된 Python 모듈 형식
    #https://stackoverflow.com/questions/41344168/what-is-a-python-layer-in-caffe
    #해당 클래스를 바탕으로 Layer를 생성하며 
    #리턴된 변수에 값을 채워넣으면 자동으로 Run된다.
    #여기서 만들어진 Class 내부에서 실질적인 databatch load가 이루어짐.


    #Glove = Global vectors for word representation
    #https://www.aclweb.org/anthology/D14-1162
    #Pretrained 된 GloveVector를 Concat에 사용.

    #img_feature는 이미 Resnet512 통과후 L2를 적용한 Preprocessing이 끝난 상태의 Feature Vector.

    n.data, n.cont, n.img_feature, n.label, n.glove = L.Python(\
        module='vqa_data_provider_layer', layer='VQADataProviderLayer', param_str=mode_str, ntop=5 )
    #module = python 파일이름
    #layer = layer형식이 맞춰진 python class
    #param_str = json으로 Data Load시 사용된 파라미터, 내부 class에 self.param_str = modestr 로 저장된다
    #ntop = 각 setup , forward backward의 top 변수의 크기


    #보통 textual Embed의 뜻은 => texture -> number
    #Embed 3000개의 Vector종류를
    #300개로 compact하게 표현함
    n.embed_ba = L.Embed(n.data, input_dim=question_vocab_size, num_output=300, \
        weight_filler=dict(type='uniform',min=-0.08,max=0.08))
    #Tanh 적용
    n.embed = L.TanH(n.embed_ba) 
    #Glove Data와 Concat
    concat_word_embed = [n.embed, n.glove]
    n.concat_embed = L.Concat(*concat_word_embed, concat_param={'axis': 2}) # T x N x 600

    # LSTM1
    n.lstm1 = L.LSTM(\
                   n.concat_embed, n.cont,\
                   recurrent_param=dict(\
                       num_output=1024,\
                       weight_filler=dict(type='uniform',min=-0.08,max=0.08),\
                       bias_filler=dict(type='constant',value=0)))
    tops1 = L.Slice(n.lstm1, ntop=T, slice_param={'axis':0})
    for i in xrange(T-1):
        n.__setattr__('slice_first'+str(i), tops1[int(i)])
        n.__setattr__('silence_data_first'+str(i), L.Silence(tops1[int(i)],ntop=0))
    n.lstm1_out = tops1[T-1]
    n.lstm1_reshaped = L.Reshape(n.lstm1_out,\
                          reshape_param=dict(\
                              shape=dict(dim=[-1,1024])))
    n.lstm1_reshaped_droped = L.Dropout(n.lstm1_reshaped,dropout_param={'dropout_ratio':0.3})
    n.lstm1_droped = L.Dropout(n.lstm1,dropout_param={'dropout_ratio':0.3})
    # LSTM2
    n.lstm2 = L.LSTM(\
                   n.lstm1_droped, n.cont,\
                   recurrent_param=dict(\
                       num_output=1024,\
                       weight_filler=dict(type='uniform',min=-0.08,max=0.08),\
                       bias_filler=dict(type='constant',value=0)))
    tops2 = L.Slice(n.lstm2, ntop=T, slice_param={'axis':0})

    #https://www.programcreek.com/python/example/107865/caffe.NetSpec 참조.
    # give top2[~] the name specified by argument `slice_second`
    #변수 부여 기능
    for i in xrange(T-1):
        n.__setattr__('slice_second'+str(i), tops2[int(i)])
        n.__setattr__('silence_data_second'+str(i), L.Silence(tops2[int(i)],ntop=0))

    #마지막 LSTM output을 사용.
    n.lstm2_out = tops2[T-1]
    n.lstm2_reshaped = L.Reshape(n.lstm2_out,\
                          reshape_param=dict(\
                              shape=dict(dim=[-1,1024])))
    n.lstm2_reshaped_droped = L.Dropout(n.lstm2_reshaped,dropout_param={'dropout_ratio':0.3})
    concat_botom = [n.lstm1_reshaped_droped, n.lstm2_reshaped_droped]
    n.lstm_12 = L.Concat(*concat_botom)


    #lstm1의 output => 1024 reshape뒤 dropout
    #lstm2의 output => 1024 reshape뒤 dropout
    #concat 

    n.q_emb_tanh_droped_resh = L.Reshape(n.lstm_12,reshape_param=dict(shape=dict(dim=[-1,2048,1,1])))
    #L.Tile 차원을 자동으로 안맞춰주므로 차원맞춤 함수. 2048,1 (tile=14, axis=1)  =>2048,14
    n.q_emb_tanh_droped_resh_tiled_1 = L.Tile(n.q_emb_tanh_droped_resh, axis=2, tiles=14)
    n.q_emb_tanh_droped_resh_tiled = L.Tile(n.q_emb_tanh_droped_resh_tiled_1, axis=3, tiles=14)

    n.i_emb_tanh_droped_resh = L.Reshape(n.img_feature,reshape_param=dict(shape=dict(dim=[-1,2048,14,14])))
    
    n.blcf = L.CompactBilinear(n.q_emb_tanh_droped_resh_tiled, n.i_emb_tanh_droped_resh, compact_bilinear_param=dict(num_output=16000,sum_pool=False))
    n.blcf_sign_sqrt = L.SignedSqrt(n.blcf)
    n.blcf_sign_sqrt_l2 = L.L2Normalize(n.blcf_sign_sqrt)
    #논문 그림과 달리 Dropout 추가 
    n.blcf_droped = L.Dropout(n.blcf_sign_sqrt_l2,dropout_param={'dropout_ratio':0.1})


    
    # multi-channel attention
    n.att_conv1 = L.Convolution(n.blcf_droped, kernel_size=1, stride=1, num_output=512, pad=0, weight_filler=dict(type='xavier'))
    n.att_conv1_relu = L.ReLU(n.att_conv1)
    #논문 그림과 달리 output dim이 2
    n.att_conv2 = L.Convolution(n.att_conv1_relu, kernel_size=1, stride=1, num_output=2, pad=0, weight_filler=dict(type='xavier'))
    n.att_reshaped = L.Reshape(n.att_conv2,reshape_param=dict(shape=dict(dim=[-1,2,14*14])))
    n.att_softmax = L.Softmax(n.att_reshaped, axis=2)
    #softmax로 attentionmap 생성
    #14x14 Softmax map이 2개 생성

    n.att = L.Reshape(n.att_softmax,reshape_param=dict(shape=dict(dim=[-1,2,14,14])))
    #두가지 att_map을 각각 Slice
    att_maps = L.Slice(n.att, ntop=2, slice_param={'axis':1})
    n.att_map0 = att_maps[0]
    n.att_map1 = att_maps[1]

    dummy = L.DummyData(shape=dict(dim=[batchsize, 1]), data_filler=dict(type='constant', value=1), ntop=1)
    n.att_feature0  = L.SoftAttention(n.i_emb_tanh_droped_resh, n.att_map0, dummy)
    n.att_feature1  = L.SoftAttention(n.i_emb_tanh_droped_resh, n.att_map1, dummy)
    n.att_feature0_resh = L.Reshape(n.att_feature0, reshape_param=dict(shape=dict(dim=[-1,2048])))
    n.att_feature1_resh = L.Reshape(n.att_feature1, reshape_param=dict(shape=dict(dim=[-1,2048])))
    n.att_feature = L.Concat(n.att_feature0_resh, n.att_feature1_resh)
    #각각 ATT를 곱한값을 연산뒤 Concat한다.

    # merge attention and lstm with compact bilinear pooling
    n.att_feature_resh = L.Reshape(n.att_feature, reshape_param=dict(shape=dict(dim=[-1,4096,1,1])))
    #그뒤 4096으로 Reshape

    n.lstm_12_resh = L.Reshape(n.lstm_12, reshape_param=dict(shape=dict(dim=[-1,2048,1,1])))
    
    #논문과 달리 가로축 세로축 inputVector크기가 다름 
    #논문 2048 2048
    #코드 4096 2048
    n.bc_att_lstm = L.CompactBilinear(n.att_feature_resh, n.lstm_12_resh, 
                                      compact_bilinear_param=dict(num_output=16000,sum_pool=False))
    #SignedSqrt
    n.bc_sign_sqrt = L.SignedSqrt(n.bc_att_lstm)
    #L2_Normalize
    n.bc_sign_sqrt_l2 = L.L2Normalize(n.bc_sign_sqrt)

    #Dropout
    n.bc_dropped = L.Dropout(n.bc_sign_sqrt_l2, dropout_param={'dropout_ratio':0.1})
    n.bc_dropped_resh = L.Reshape(n.bc_dropped, reshape_param=dict(shape=dict(dim=[-1, 16000])))

    #FullyConnected
    n.prediction = L.InnerProduct(n.bc_dropped_resh, num_output=3000, weight_filler=dict(type='xavier'))
    
    n.loss = L.SoftmaxWithLoss(n.prediction, n.label)
    return n.to_proto()

def make_answer_vocab(adic, vocab_size):
    """
    Returns a dictionary that maps words to indices.
    """
    adict = {'':0}
    nadict = {'':1000000}
    vid = 1
    for qid in adic.keys():
        answer_obj = adic[qid]
        answer_list = [ans['answer'] for ans in answer_obj]
        
        for q_ans in answer_list:
            # create dict
            if adict.has_key(q_ans):
                nadict[q_ans] += 1
            else:
                nadict[q_ans] = 1
                adict[q_ans] = vid
                vid +=1

    # debug
    nalist = []
    for k,v in sorted(nadict.items(), key=lambda x:x[1]):
        nalist.append((k,v))

    # remove words that appear less than once 
    n_del_ans = 0
    n_valid_ans = 0
    adict_nid = {}
    for i, w in enumerate(nalist[:-vocab_size]):
        del adict[w[0]]
        n_del_ans += w[1]
    for i, w in enumerate(nalist[-vocab_size:]):
        n_valid_ans += w[1]
        adict_nid[w[0]] = i
    
    return adict_nid

def make_question_vocab(qdic):
    """
    Returns a dictionary that maps words to indices.
    """
    vdict = {'':0}
    vid = 1
    for qid in qdic.keys():
        # sequence to list
        q_str = qdic[qid]['qstr']
        q_list = VQADataProvider.seq_to_list(q_str)

        # create dict
        for w in q_list:
            if not vdict.has_key(w):
                vdict[w] = vid
                vid +=1

    return vdict

def make_vocab_files():
    """
    Produce the question and answer vocabulary files.
    """
    print 'making question vocab...', config.QUESTION_VOCAB_SPACE
    qdic, _ = VQADataProvider.load_data(config.QUESTION_VOCAB_SPACE)
    question_vocab = make_question_vocab(qdic)
    print 'making answer vocab...', config.ANSWER_VOCAB_SPACE
    _, adic = VQADataProvider.load_data(config.ANSWER_VOCAB_SPACE)
    answer_vocab = make_answer_vocab(adic, config.NUM_OUTPUT_UNITS)
    return question_vocab, answer_vocab

def main():
    #Make Result Fold
    if not os.path.exists('./result'):
        os.makedirs('./result')

    #Make Empty Dict Variable
    question_vocab, answer_vocab = {}, {}
    if os.path.exists('./result/vdict.json') and os.path.exists('./result/adict.json'):
        print 'restoring vocab'
        with open('./result/vdict.json','r') as f:
            question_vocab = json.load(f)
        with open('./result/adict.json','r') as f:
            answer_vocab = json.load(f)
    else:
        #Load TrainData
        question_vocab, answer_vocab = make_vocab_files()
        with open('./result/vdict.json','w') as f:
            json.dump(question_vocab, f)
        with open('./result/adict.json','w') as f:
            json.dump(answer_vocab, f)
        #json.dump Encoding to string object

    print 'question vocab size:', len(question_vocab)
    print 'answer vocab size:', len(answer_vocab)


    #################Read ConfigFile + Make Network##############################
    with open('./result/proto_train.prototxt', 'w') as f:
        f.write(str(qlstm(config.TRAIN_DATA_SPLITS, config.BATCH_SIZE, \
            config.MAX_WORDS_IN_QUESTION, len(question_vocab))))
    
    with open('./result/proto_test.prototxt', 'w') as f:
        f.write(str(qlstm('val', config.VAL_BATCH_SIZE, \
            config.MAX_WORDS_IN_QUESTION, len(question_vocab))))


    #######################################################################

    #당시 caffe -python 버전은 multi gpu지원하지않음 c++interface only
    #현재는 python도 지원
    caffe.set_device(config.GPU_ID)
    caffe.set_mode_gpu()
    solver = caffe.get_solver('./qlstm_solver.prototxt')
    #solver.prototxt 학습 config를 Load할때 사용.
    
    train_loss = np.zeros(config.MAX_ITERATIONS)
    results = []

    for it in range(config.MAX_ITERATIONS):
        solver.step(1)
        #step : forward backward update 3가지를 실시한다.

        # store the train loss
        train_loss[it] = solver.net.blobs['loss'].data
   
        if it % config.PRINT_INTERVAL == 0:
            print 'Iteration:', it
            c_mean_loss = train_loss[it-config.PRINT_INTERVAL:it].mean()
            print 'Train loss:', c_mean_loss
        if it != 0 and it % config.VALIDATE_INTERVAL == 0:
            solver.test_nets[0].save('./result/tmp.caffemodel')
            print 'Validating...'
            test_loss, acc_overall, acc_per_ques, acc_per_ans = exec_validation(config.GPU_ID, 'val', it=it)
            print 'Test loss:', test_loss
            print 'Accuracy:', acc_overall
            results.append([it, c_mean_loss, test_loss, acc_overall, acc_per_ques, acc_per_ans])
            best_result_idx = np.array([x[3] for x in results]).argmax()
            print 'Best accuracy of', results[best_result_idx][3], 'was at iteration', results[best_result_idx][0]
            drawgraph(results)

#메인 파이썬으로 실행될때만 fun_main실행
if __name__ == '__main__':
    main()
