import os
from subprocess import call

trandir = "transcripts/"
dir = "../../../data/KB-2k/"
sublist = next(os.walk(dir))[1]
for i,subdir in enumerate(sublist):
    print(i)
    subdir = dir+subdir+'/'
    audiodir = subdir+"audio/"
    textdir = subdir+"text/"
    if (not os.path.exists(textdir)):
        os.makedirs(textdir)
    for wavfile in os.listdir(audiodir):
        id = wavfile[:-4]
        tranfile = trandir+id+'.json'
        wavfile = audiodir+wavfile
        outfile = textdir+id+'.TextGrid'
        command = "python ../p2fa-vislab/align.py --textgrid {} {} {}".format(wavfile,tranfile,outfile)
        call(command,shell=True)
