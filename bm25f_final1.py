#!/usr/bin/env python
# coding: utf-8
from bs4 import BeautifulSoup as BS
import io
import os
import BM25F.core
import BM25F.en
import BM25F.exp
import numpy as np
import pandas as pd
tokenizer = BM25F.en.Tokenizer(token_filter=BM25F.en.TokenFilter())

bj = BM25F.exp.bag_jag()
query_file = open('/home/fanyimeng/queries.txt')
query_txt=query_file.readlines()
def dcg_at_k(r, k, method=0):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):

    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max
d=pd.DataFrame(index=[5,7,11,23,27,33,41,45,51,53,55,57,63,65,67,71,73,79,81,89,93,95],columns=[0])
ndcgs=[]
for query_id in [5,7,11,23,27,33,41,45,51,53,55,57,63,65,67,71,73,79,81,89,93,95]:
    path='/home/fanyimeng/documents/'+str(query_id)
    list_dirs=os.walk(path)
    l=[]
    count=0
    for root, dirs, files in list_dirs:
        for f in files:
            count+=1
            label_id=root[-1]
            with io.open(os.path.join(root, f), encoding='utf-8') as page:
                html = BS(page, 'lxml')
            st=''
            sb=''
            sa=''
            sli=''
            sh1=''
            sh2=''
            sbo=''
            sst=''
            std=''
            sf=''
            for i in html.find_all('title'):
                st += (i.get_text() + ' ')
            for i in html.find_all('body'):
                sb += (i.get_text() + ' ')
            for i in html.find_all('a'):
                sa += (i.get_text() + ' ')
            for i in html.find_all('li'):
                sli += (i.get_text() + ' ')
            for i in html.find_all('h1'):
                sh1 += (i.get_text() + ' ')
            for i in html.find_all('h2'):
                sh2 += (i.get_text() + ' ')
            for i in html.find_all('b'):
                sbo += (i.get_text() + ' ')
            for i in html.find_all('strong'):
                sst += (i.get_text() + ' ')
            for i in html.find_all('td'):
                std += (i.get_text() + ' ')
            for i in html.find_all('font'):
                sf += (i.get_text() + ' ')
            bd0 = BM25F.exp.bag_dict().read(tokenizer, {
                '_id': label_id,
                'title': st,
                'body':sb,
                'a':sa,
                'li':sli,
                'h1':sh1,
                'h2':sh2,
                # 'b':sbo,
                # 'strong':sst,
                'td':std,
                # 'font':sf,
            })
            l.append(bd0)
            bj.append(bd0)



    query = BM25F.exp.bag_of_words().read(tokenizer, query_txt[query_id - 1].strip('\n'))

    boost = BM25F.core.param_dict(default=1.0)
    # FINAL
    boost['title']=50
    boost['body']=1
    boost['a']=20
    boost['li']=15
    boost['h1']=50
    boost['h2']=50
    # boost['b']=30
    # boost['strong']=30
    boost['td']=5
    # boost['font']=40
    k1 = 2.0

    b = BM25F.core.param_dict(default=0.75)
    b['title'] = 0.50
    b['body'] = 1.00
    depth = count

    scorer = BM25F.core.batch('_id', query, bj, boost, k1, b)

    print(query_id, ':', scorer.top(depth, l), ndcg_at_k(scorer.top(depth, l), 5))
    d.loc[query_id, 0] = scorer.top(depth, l)
    ndcgs.append(ndcg_at_k(scorer.top(depth, l), 5))
d.to_pickle('/home/fanyimeng/FINAL.pkl')
print(sum(ndcgs) / len(ndcgs))


