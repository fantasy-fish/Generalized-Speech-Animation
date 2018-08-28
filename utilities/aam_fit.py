import textgrid
import os
import menpo.io as mio
from menpofit.aam import LucasKanadeAAMFitter, WibergInverseCompositional
from menpodetect import load_dlib_frontal_face_detector
import numpy as np
import time

def process_one_sentence(video,fitter,detector):
    shapes_list = list()
    images = [i for i in os.listdir(video) if i!="aam"]
    images = sorted(images)
    images = [video+img for img in images]
    for i,imgname in enumerate(images):
        #print(i)
        img = mio.import_image(imgname)
        img = img.as_greyscale()
        bboxes = detector(img)# Detect
        initial_bbox = bboxes[0]# initial bbox
        result = fitter.fit_from_bb(img, initial_bbox, max_iters=15)
        shape_parameter = result.shape_parameters[-1][4:]#The first four are for global tramsformations
        shapes_list.append(shape_parameter)
    np.savetxt(video+"aam", shapes_list, fmt="%.8g")


def process_data(dir):
    dirlist = os.listdir(dir)
    dirlist = [d for d in dirlist if not os.path.isfile(os.path.join(dir, d))]
    dirlist = sorted(dirlist)
    # load the aam model
    aam = mio.import_pickle("aam.pkl")
    # create fiiter
    fitter = LucasKanadeAAMFitter(aam, lk_algorithm_cls=WibergInverseCompositional,
                                  n_shape=16, n_appearance=104)
    # Load detector
    detector = load_dlib_frontal_face_detector()
    #load the sentences
    for j,subdir in enumerate(dirlist):
        ids = os.listdir(os.path.join(dir,subdir,"video/"))
        ids = [i for i in ids if "head" not in i]
        ids = sorted(ids)
        for k,id in enumerate(ids):
            t= time.time()
            video = os.path.join(dir,subdir,"video",id+'/')
            process_one_sentence(video,fitter,detector)
            print(j, k, time.time()-t)

if __name__ == "__main__":
    data_dir = "../../../data/KB-2k/"
    process_data(data_dir)

