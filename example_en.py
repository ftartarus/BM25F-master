#!/usr/bin/env python
# coding: utf-8
from bs4 import BeautifulSoup as BS
import io
import BM25F.core
import BM25F.en
import BM25F.exp

tokenizer = BM25F.en.Tokenizer(token_filter=BM25F.en.TokenFilter())

bj = BM25F.exp.bag_jag()

h3='/Users/Fan/Desktop/data/research/documents/8/4/clueweb12-0700wb-27-24140.html'
h1='/Users/Fan/Desktop/data/research/documents/8/0/clueweb12-0004wb-05-07573.html'
with io.open(h3, encoding='utf-8') as page:
    HTML4 = BS(page, 'lxml')
with io.open(h1, encoding='utf-8') as page:
    HTML1 = BS(page, 'lxml')
s3t=''
s3p=''
s3a=''
s3li=''
s3td=''
for i in HTML4.find_all('title'):
    s3t+=(i.get_text()+' ')
for i in HTML4.find_all('p'):
    s3p+=(i.get_text()+' ')
for i in HTML4.find_all('a'):
    s3a+=(i.get_text()+' ')
for i in HTML4.find_all('li'):
    s3li+=(i.get_text()+' ')
for i in HTML4.find_all('td'):
    s3td+=(i.get_text()+' ')

s1t=''
s1p=''
s1a=''
s1li=''
s1td=''
for i in HTML1.find_all('title'):
    s1t+=(i.get_text()+' ')
for i in HTML1.find_all('p'):
    s1p+=(i.get_text()+' ')
for i in HTML1.find_all('a'):
    s1a+=(i.get_text()+' ')
for i in HTML1.find_all('li'):
    s1li+=(i.get_text()+' ')
for i in HTML1.find_all('td'):
    s1td+=(i.get_text()+' ')

with io.open(h1, encoding='utf-8') as page:
    HTML1 = BS(page, 'lxml')
tags=['p','title','a','li','td']

bd0 = BM25F.exp.bag_dict().read(tokenizer, {
    '_id': '4',
    'title': s3t,
    'p': s3p,
    # 'a':s3a,
    'li':s3li,
    # 'td':s3td,
    # 'body':s3b,

})
bj.append(bd0)

bd1 = BM25F.exp.bag_dict().read(tokenizer, {
    '_id': '0',
    'title': s1t,
    'p': s1p,
    # 'a':s1a,
    'li':s1li,
    # 'td':s1td,
})
bj.append(bd1)

bd2 = BM25F.exp.bag_dict().read(tokenizer, {
    '_id': '2',
    'title': 'he love',
    'td': 'he loves you',
})
bj.append(bd2)

bd3 = BM25F.exp.bag_dict().read(tokenizer, {
    '_id': '3',
    'title': 'love',
    'td': 'he loves you',
})
bj.append(bd3)

query = BM25F.exp.bag_of_words().read(tokenizer, 'World Table Tennis Championships')

boost = BM25F.core.param_dict(default=1.0)
boost['title']=100
boost['p']=0.1
boost['li']=1

k1 = 2.0

b = BM25F.core.param_dict(default=0.75)
b['title'] = 0.50
b['p'] = 1.00


scorer = BM25F.core.batch('_id', query, bj, boost, k1, b)

print(scorer.top(2, [bd0, bd1]))
