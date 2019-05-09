import io
import os
import numpy as np
import tensorflow as tf
from librosa import effects

from utils import *

from train import Graph
from hyperparams import Hyperparams as hp
import tqdm
import re

def load_vocab():
    char2idx = {char: idx for idx, char in enumerate(hp.vocab)}
    idx2char = {idx: char for idx, char in enumerate(hp.vocab)}
    return char2idx, idx2char

SEC_PER_CHAR = float(10)/180  #[sec/char]
SEC_PER_ITER = float(12)/200  #[sec/iter]

def get_EOS_index(text):
    # text : texto codificado en numeros
    # Load vocab
    char2idx, idx2char = load_vocab()

    _text = np.array([idx2char[t] for t in text])
    return np.argmax(_text == hp.EOS_char)

def get_EOS_fire(alignment,text):
    EOS_index = get_EOS_index(text)
    text_max_indicies = np.argmax(alignment,axis = 0)
    r = []
    for i,max_index in enumerate(text_max_indicies):
        if max_index == EOS_index:
            r.append(i)
    if not len(r) == 0:
        return max(r)
    return None

import scipy
def save_wav(wav, path, sr):
  wav *= 32767 / max(0.01, np.max(np.abs(wav)))
  scipy.io.wavfile.write(path, sr, wav.astype(np.int16))

def wahio(n):
    lista=list(str(n))
    inverse,new,con,=lista[::-1],['','','','','','','','',''],0
    for i in inverse:
        new[con]=int(i)
        con+=1
    a,b,c,d,e,f,g,h,i=new[::-1]
    if len(str(new[3]))>0:
        new.insert(3,'.')
    if len(str(new[7]))>0:
        new.insert(7,'.')
    numero=new[::-1]
    #print('\n'+'Resultado para: ',end='')
 
    unidad={1:'un', 2:'dos', 3:'tres', 4:'cuatro', 5:'cinco', 6:'seis', 7:'siete', 8:'ocho', 9:'nueve',0:'','':''}
    unidadi={1:'uno', 2:'dos', 3:'tres', 4:'cuatro', 5:'cinco', 6:'seis', 7:'siete', 8:'ocho', 9:'nueve',0:'','':''}
    unidad2={10:'diez', 11:'once', 12:'doce', 13:'trece', 14:'catorce', 15:'quince',16:'diez y seis',17:'diez y siete', 18:'diez y ocho', 19:'diez y nueve'}
    decena={1:'diez', 2:'veinti', 3:'treinta', 4:'cuarenta', 5:'cincuenta', 6:'sesenta', 7:'setenta', 8:'ochenta', 9:'noventa','':'',0:''}
    centena={1:'ciento', 2:'dos cientos',3:'tres cientos',4:'cuatro cientos',5:'quinientos',6:'seis cientos',7:'setecientos',8:'ocho cientos',9:'novecientos','':'',0:''}
 
    a=centena[a]
    if b==1 and c<6:
        b,c=unidad2[int(str(b)+str(c))],'millones'
    elif c==1:
        c,b='un millon',decena[b]
    elif b==0:
        b,c='',(unidad[c]+len(str(c))*' millones')
    else:
        b=(decena[b]+len(str(b))*' y')
        c=(unidad[c]+len(str(c))*' millones')
    d=centena[d]
    if e==1 and f<6:
        e,f=unidad2[int(str(e)+str(f))],'mil'
    elif f==0:
        e,f=decena[e],'mil'
    elif e==0:
        e,f='',(unidad[f]+len(str(f))*' mil')
    else:
        e=(decena[e]+len(str(e))*' y')
        f=(unidad[f]+len(str(f))*' mil')
    g=centena[g]
    if h==1 and i<6:
        h,i=unidad2[int(str(h)+str(i))],''
    elif h==0:
        h,i='',unidadi[i]
    else:
        if i==0:
            i,h='',decena[h]
        else:
            i,h=unidadi[i],decena[h]+len(str(h))*' y'
    orden=[a,b,c,d,e,f,g,h,i]
    cadena = ' '.join(orden)
    return cadena.strip()





class Synthesizer:
  def load(self, model_name='tacotron'):
    # Load graph
    self.g = Graph(mode="synthesize"); print("Graph loaded")

    saver = tf.train.Saver()

    self.session = tf.Session()
    self.session.run(tf.global_variables_initializer())

    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Text2Mel')
    saver1 = tf.train.Saver(var_list=var_list)
    saver1.restore(self.session, tf.train.latest_checkpoint(hp.logdir + "-1"))
    print("Text2Mel Restored!")

    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'SSRN') + \
               tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'gs')
    saver2 = tf.train.Saver(var_list=var_list)
    saver2.restore(self.session, tf.train.latest_checkpoint(hp.logdir + "-2"))
    print("SSRN Restored!")

  def synthesize(self, text):
    text = text.strip() +'.'
    char2idx, idx2char = load_vocab()
    
    ##  Convertir numeros a palabras
    lista_numeros = re.findall(r'\d+', text)
    for num in lista_numeros:
        text = text.replace(num,wahio(num))


    print('texto : ', text)
    
    texts = np.zeros((1, hp.max_N), np.int32)
    texts[0,:len(text)] = [char2idx[char] for char in text]

    num_chars = len(text)

    ## mel
    Y = np.zeros((1,hp.max_T, hp.n_mels), np.float32)
    prev_max_attentions = np.zeros((1,), np.int32)
    for j in tqdm.tqdm(range(2*num_chars)):
        _gs, _Y, _max_attentions, _alignments = \
            self.session.run([self.g.global_step, self.g.Y, self.g.max_attentions, self.g.alignments],
                     {self.g.L: texts,
                      self.g.mels: Y,
                      self.g.prev_max_attentions: prev_max_attentions})
        Y[:, j, :] = _Y[:, j, :]
        prev_max_attentions = _max_attentions[:, j]

    ## mag
    Z = self.session.run(self.g.Z, {self.g.Y: Y})

    wav = spectrogram2wav(Z[0,:,:])

    out = io.BytesIO()
    save_wav(wav, out, hp.sr)
    return out.getvalue()



