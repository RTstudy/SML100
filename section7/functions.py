#!/usr/bin/env python
# coding: utf-8

# # 関数定義

# ## 損失関数

# In[2]:

import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import scipy
from scipy import stats
from numpy.random import randn
from numpy.random import randint
import copy


##　二乗誤差
def sq_loss(y):
    if len(y) == 0:
        return 0
    else:
        y_bar = np.mean(y)
        return np.linalg.norm(y-y_bar)**2  # L2ノルムを計算し二乗している


# In[3]:


## 頻度
### 各要素の頻度を算出
def freq(y):
    y = list(y)
    return [y.count(i) for i in set(y)]


# In[4]:


## 最頻値
def mode(y):
    n = len(y)
    if n == 0:
        return 0
    return max(freq(y))


# In[5]:


# 誤り数
def mis_match(y):
    return len(y) - mode(y)


# In[6]:


# Gini
def gini(y):
    n = len(y)
    if n == 0:
        return 0
    fr = freq
    return sum([fr[i]/n * (n - fr[i]) for i in range(len(fr))])


# In[7]:


# Entropy
def entropy(y):
    n = len(y)
    if n == 0:
        return 0
    freq = [y.count(i) for i in set(y)]
    return np.sum([-freq[i] * np.log(freq[i]/n) for i in range(len(freq))])


# In[8]:


def mode_max(y):
    if len(y) == 0:
        return -np.inf
    count = np.bincount(y)
    return np.argmax(count)


# ## 実行用関数


# In[13]:


# 枝を伸ばす
def dt(x, y, alpha = 0, n_min = 1, rf = 0):
    # x:入力データ
    # y:ターゲットデータ
    # alpha:後で利用
    # n_min:各ノードにおけるデータ集合の最小値
    # rf:ランダムフォレストを実行する際に利用する変数の数
    if rf == 0:
        m = x.shape[1]
    
    stack = [Stack(0, list(range(x.shape[0])), f(y))]   # stackを初期化
    node = []   # nodeを初期化
    k = -1      # kを初期化
    
    # stackに何かしらデータが残っている間のループ
    while len(stack) > 0:
        popped = stack.pop()   # stackの最後のデータを取り出してpoppedに格納
        #print(popped.parent, popped.score, popped.set)

        k = k + 1
        i, j, left, right, score, left_score, right_score = branch(x, y, popped.set, rf)  # 枝の分割を計算 Sにはpopped.setを入れる
        if popped.score - score < alpha or len(popped.set) < n_min or len(left) == 0 or len(right) == 0:
            # 以下の条件を満たすとき、枝をそこで止める
            ## 1. 一回前とのスコアの差分が閾値alphaより小さい
            ## 2. ノードのデータ集合数がn_minより小さい
            ## 3. 左の枝が空集合
            ## 4. 右の枝が空集合
            node.append(Node(popped.parent, -1, 0, popped.set))
        else:
            node.append(Node(popped.parent, j, x[i,j], popped.set))
            stack.append(Stack(k, right, right_score))
            stack.append(Stack(k, left, left_score))
    ### 枝を伸ばす処理はここまで
    
    # ノードの分岐構造をデータにする
    for h in range(k, -1, -1):
        # kは枝の数
        # 各ノードにleft, rightを設定する、そのための初期化
        node[h].left = 0
        node[h].right = 0

    for h in range(k, 0, -1):  # kから降順で最後の値の一つ手前まで
        pa = node[h].parent       # node[h]の親ノードをpaに格納
        #print(h, pa)
        if node[pa].right == 0:   # node[h]の親ノードの右側にまだ値が格納されていない場合
            node[pa].right = h   # node[h]をその親ノードのrightに格納
        else:                     # そうでない場合
            node[pa].left = h    # node[h]をその親ノードのleftに格納
        #print(h, pa, node[pa].right, node[pa].left)
    
    # 
    if f == sq_loss:      # 損失関数を二乗誤差にした場合の処理
        g = np.mean       # 平均値を引数に与える関数とする
    else:
        g = mode_max      # 中央値を引数に与える関数とする
    for h in range(k+1):
        if node[h].j == -1:    # ノードが枝分かれしていない場合
            node[h].center = g(y[node[h].set])    # そのノードの集合におけるターゲットの平均もしくは最頻値をそのノードのcenterとする
        else:
            node[h].center = 0                    # ノードが枝分かれしている場合はcenterを決めずとする
            
    return node


# ## データ保持用class定義

# In[11]:


# データスタック
    # データ格納用にクラスを利用
class Stack:
    def __init__(self, parent, set, score):
        self.parent = parent   # 親ノードの番号
        self.set = set         # データのindexリスト
        self.score = score     # 損失関数


# In[12]:


# ノード
    # データ格納用にクラスを利用
class Node:
    def __init__(self, parent, j, th, set):
        # parent:親ノードの番号
        # j:自身のノード番号
        # th:分岐の際の閾値
        # set:ノード内のデータ集合のindexリスト
        self.parent = parent
        self.j = j
        self.th = th
        self.set = set


# ## 評価用関数

# In[10]:


# 計算済みのモデルを使って新しいデータに対する予測値を計算
def value(u, node):
    # u:入力データ
    # node:計算済みのモデル
    r = 0    # 初期化
    
    #print(u)
    
    # 端点に達するまでループ
    while node[r].j != -1:
        #print(r)
        #print(node[r].left, node[r].right)
        if u[node[r].j] < node[r].th:   # 入力値が閾値よりも小さい場合、左のノードへ
            #print(node[r].left)
            #print(u[node[r].j], node[r].th)
            r = node[r].left
        else:                           # 入力値が閾値以上の場合、右のノードへ
            #print(node[r].right)
            r = node[r].right
    return node[r].center               # 端点に達した時、端点が格納しているcenterの値を返す


# ## 描画用関数

# In[ ]:


from igraph import *
def draw_graph(node):
    r = len(node)
    col = []
    for h in range(r):
        col.append(node[h].j)
    colorlist = ['#ffffff','#fff8ff','#fcf9ce','#d6fada','#d7ffff','#d9f2f8','#fac8be','#ffebff','#ffffe0','#fdf5e6','#fac8be','#f8ecd5','#ee82ee']
    color = [colorlist[col[i]] for i in range(r)]
    edge = []
    for h in range(1,r):
        edge.append([node[h].parent,h])
        g = Graph(edges = edge, directed = True)
        layout = g.layout_reingold_tilford(root = [0])
    out = plot(g,
               vertex_size = 15,
               layout = layout,
               bbox = (300, 300),
               vertex_label = list(range(r)),
               vertex_color = color)
    return out

