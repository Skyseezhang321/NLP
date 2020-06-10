from django.shortcuts import render
from django.shortcuts import HttpResponse
import torch
import numpy as np
import jieba
import os
from textGen.text_generation import RNNModule

# Create your views here.
#视图函数处理用户请求，也就是编写业务处理逻辑，一般都在views.py文件里
def predict(device, net, words, n_vocab, vocab_to_int, int_to_vocab, top_k=5):
    net.eval()

    state_h, state_c = net.zero_state(1)
    state_h = state_h.to(device)
    state_c = state_c.to(device)
    for w in words:
        ix = torch.tensor([[vocab_to_int[w]]]).to(device)
        output, (state_h, state_c) = net(ix, (state_h, state_c))

    _, top_ix = torch.topk(output[0], k=top_k)
    choices = top_ix.tolist()
    choice = np.random.choice(choices[0])

    words.append(int_to_vocab[choice])

    for _ in range(100):
        ix = torch.tensor([[choice]]).to(device)
        output, (state_h, state_c) = net(ix, (state_h, state_c))

        _, top_ix = torch.topk(output[0], k=top_k)
        choices = top_ix.tolist()
        choice = np.random.choice(choices[0])
        words.append(int_to_vocab[choice])
    return ' '.join(words)


def preprocess_text(content_lines):
    try:
        segs = jieba.lcut(content_lines)
        segs = ' '.join(segs)
        return segs
    except Exception as e:
        return ' '


res_list = []

def text_gen_pro(initial_words):
    # 预测
    predict_params = torch.load(
        '/Users/exudingtao/PycharmProjects/Exuding-NLP/text_genaration_rnn/djangopro/'
        'textGen/checkpoint_pt/net_predict_params.pkl')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_vocab = predict_params['n_vocab']
    vocab_to_int = predict_params['vocab_to_int']
    int_to_vocab = predict_params['int_to_vocab']

    # model = torch.load(
    #     '/Users/exudingtao/PycharmProjects/Exuding-NLP/text_genaration_rnn/djangopro/textGen/'
    #     'checkpoint_pt/model-text-generation.pth')

    model = RNNModule(n_vocab, 32, 64, 64)
    model.load_state_dict(torch.load('/Users/exudingtao/PycharmProjects/Exuding-NLP/'
                                     'text_genaration_rnn/djangopro/textGen/'
                                     'checkpoint_pt/model-1000.pth'))

    text_gen = predict(device, model, initial_words, n_vocab,
                       vocab_to_int, int_to_vocab, top_k=5)
    return text_gen


def index(request):
    if request.method == 'POST':
        text_gen_pre = request.POST.get('text_gen')
        res = preprocess_text(text_gen_pre)
        text_gen_pre_list = res.split()
        if text_gen_pre_list:
            initial_words = text_gen_pre_list
        else:
            initial_words = ['I', 'am']
        text_gen = text_gen_pro(initial_words)
        temp = {'text_gen_done': text_gen}
        res_list.append(temp)
    return render(request, 'index.html', {'data': res_list})



