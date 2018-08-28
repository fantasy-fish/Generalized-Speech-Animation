import textgrid
import os
import numpy as np
import time

def load_textgrid(file):
    t = textgrid.TextGrid()
    t.read(file)
    l = t.getFirst('phone')
    return l

def clean_phoneme(str):
    str = str.lower()
    str = ''.join([i for i in str if not i.isdigit()])
    return str

def change_x_to_onehot(x,class_set):
    n_class = len(class_set)
    y_onehot_mapping = dict(zip(class_set, range(n_class)))
    onehot = []
    for label in x:
        tmp = [0] * n_class
        tmp[y_onehot_mapping[label]] = 1
        onehot.append(tmp)
    return np.hstack(onehot)

def load_one_sentence(text,video,normalize=True,clean=True,first=False):
    phonemes_list = list()
    phonemes = load_textgrid(text)
    phonemes_list_no = [i.mark for i in phonemes] #no stands for non-overlapping
    phonemes_list_no = [p for p in phonemes_list_no if p!='']
    if clean:
        phonemes_list_no = map(clean_phoneme,phonemes_list_no)#clean the phonemes
    maxlength = phonemes.maxTime
    shapes = np.loadtxt(video+"aam")
    if not first:
        shapes = shapes[:,1:]
    shape_clean = list()
    n_images = len(shapes)
    for i in range(n_images):
        t = maxlength*i/n_images
        phoneme = phonemes[phonemes.indexContaining(t)].mark
        if phoneme == '':
            continue
        if clean:
            phoneme = clean_phoneme(phoneme)#clean the phonemes
        phonemes_list.append(phoneme)
        if normalize:
            shape_clean.append(shapes[i]/np.max(abs(shapes[i])))
        else:
            shape_clean.append(shapes[i])
    return phonemes_list,shape_clean,phonemes_list_no

def load_data(dir,Kx,Ky,normalize=True,clean=True,first=False):
    if(Kx<Ky):
        print("Kx should be greater than Ky")
    if((Ky-Kx)%2!=0):
        print("Ky-Kx should be even")
    dirlist = os.listdir(dir)
    dirlist = [d for d in dirlist if not os.path.isfile(os.path.join(dir, d))]
    data_x = list()
    data_y = list()
    phonemes_list = list()
    #y_ = list()
    #load the sentences
    for subdir in dirlist:
        ids = os.listdir(os.path.join(dir,subdir,"video/"))
        ids = [i for i in ids if "head" not in i]
        for id in ids:
            text = os.path.join(dir,subdir,"text",id+".TextGrid")
            video = os.path.join(dir,subdir,"video",id+'/')
            x,y,phonemes = load_one_sentence(text,video,normalize,clean,first)
            #y_.append(y)
            phonemes_list += phonemes
            #create the overlapping sequences
            if(len(x)!=len(y)):
                print("What's wrong")
            y_beg = (Kx-Ky)//2
            for x_beg in range(len(x)-Kx):
                x_seq = x[x_beg:x_beg+Kx]
                y_seq = y[y_beg:y_beg+Ky]
                y_seq = np.hstack(y_seq)
                y_beg += 1
                data_x.append(x_seq)
                data_y.append(y_seq)
    #change x to one-hot vector
    phonemes_set = set(phonemes_list)
    for i in range(len(data_x)):
        x_seq = data_x[i]
        data_x[i] = change_x_to_onehot(x_seq,phonemes_set)
    print("Data loaded")
    #calculate the std for each dimension
    #y_ = np.vstack(y_)
    #print(np.std(y_,axis=0))
    #save the phonemes_set
    with open("utilities/phonemes",'w') as f:
        for ph in phonemes_set:
            f.write(ph+'\n')
    return data_x,data_y

def load_test_data(dir,Kx=11):
    phonemes = load_textgrid(dir)
    maxtime = float(phonemes.maxTime)
    #sampled at 25th frame rate
    time_samples = np.arange(0.04,maxtime,0.04)
    phoneme_samples = [phonemes[phonemes.indexContaining(t)].mark for t in time_samples]
    phoneme_samples = [clean_phoneme(ph) for ph in phoneme_samples if ph!=' ']
    x = list()
    for i in range(len(phoneme_samples)-Kx+1):
        x.append(phoneme_samples[i:i+Kx])
    with open('utilities/phonemes') as f:
        phoneme_set = f.readlines()
        phoneme_set = [ph[:-1] for ph in phoneme_set]

        phoneme_set = set(phoneme_set)
    x = [change_x_to_onehot(ii,phoneme_set) for ii in x]
    return x

def average_y(pred,Ky,n_shape):
    pred_avg=list()
    for p in pred:
        tmp=list()
        for i in range(Ky):
            tmp.append(p[i*n_shape:(i+1)*n_shape])
        tmp=np.array(tmp)
        tmp = np.mean(tmp,axis=0)
        pred_avg.append(tmp)
    return pred_avg

if __name__ == "__main__":
    #base_dir = "../../data/KB-2k/fcmh0/"
    #text = base_dir + "text/sa1.TextGrid"
    #video = base_dir + "video/sa1/"
    #load_one_sentence(text, video)
    #data_dir = "../../data/KB-2k/"
    #t =time.time()
    #load_data(data_dir,11,5)
    #print(time.time()-t)
    dir = "test/sx383.TextGrid"
    load_test_data(dir)