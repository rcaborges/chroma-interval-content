import time
import math
import numpy as np
import pandas as pd
from datetime import datetime
import sys
import os

from stat import *
import json
import os.path

def get_data(file_name):
    data = pd.read_csv(file_name, sep="\t" , quotechar="\"",engine='python', header=None)   
    return data

def get_genres(file_name):
    data = pd.read_csv(file_name, sep="\t" , quotechar="\"",engine='python', header=None)
    cols = ['ID','FILE','ARTIST','TITLE','TIME','BPM','YEAR','GENRE','DISC-TRACK','DETAILS']
    data.columns = cols
    data = data.drop('FILE', axis=1)
    data.set_index('ID')
    return data

def calc_onsets(tim,sr):
    onsets = []
    for onst in tim:
        try:
            ost = float(onst)
            #song[int(sr*ost)] = 1
            onsets.append(np.ceil(int(sr*ost)/512))
        except:
            continue
    return onsets

def calc_din(crm_arr):
    crmip = np.zeros(12)
    tmtx = []
    last_chroma = np.zeros(crm_arr.shape[1])
    for crmi in crm_arr:
        delta = []
        for i in range(12):
            delta.append(np.linalg.norm(np.roll(crmi,i) - last_chroma))
        nabla = np.max(delta) - delta
        nabla = (nabla-np.min(nabla))/(np.max(nabla)-np.min(nabla))
        #nabla = nabla/np.linalg.norm(nabla)
        if not np.isnan(nabla).any():
            tmtx.append(nabla)
        #tmtx.append(row_d + np.max(abs(np.array(row_d))))
        last_chroma = crmi
    tmtx = np.array(tmtx)
    print(tmtx.shape)
    tmtx = tmtx[1:-1,:].T
    #plot_feat(tmtx)
    tmtx = tmtx.tolist()
    return tmtx

def calc_sic(crm_arr):
    crmip = np.zeros(12)
    tmtx = []
    for crmi in crm_arr:
        row_d = []
        for d in np.arange(-5,7,1):
            sum_crm = 0
            for i in range(12):
                sum_crm = sum_crm + (crmip[i]*crmi[(i+d)%12])
            row_d.append(sum_crm)
        #row_d = row_d/np.linalg.norm(row_d)
        row_d = (row_d-np.min(row_d))/(np.max(row_d)-np.min(row_d))
        crmip = crmi
        if not np.isnan(row_d).any():
            tmtx.append(row_d)
    tmtx = np.array(tmtx)
    print(tmtx.shape)
    tmtx = tmtx[1:-1,:].T
    #plot_matrix(tmtx)
    tmtx = tmtx.tolist()
    return tmtx

def extract_librosa(file_name,tim):
    import librosa
    y, sr = librosa.load(file_name)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    onset_frames = calc_onsets(tim,sr)
    print(chroma.shape)
    trans = []
    #PRIMEIRA TRANSICAO
    #trans.append(np.sum(chroma[:,:int(onset_frames[0])],axis=1)/int(onset_frames[0]))
    for ost in range(len(onset_frames)-1):
        trans.append(np.sum(chroma[:,int(onset_frames[ost]):int(onset_frames[ost+1])],axis=1)/(int(onset_frames[ost+1])-int(onset_frames[ost])))
    #ULTIMA TRANSICAO
    trans.append(np.sum(chroma[:,int(onset_frames[-1]):],axis=1)/(chroma.shape[1] - int(onset_frames[-1])))
    trans = np.array(trans)
    print(trans.shape)
    return trans

def plot_sig(tim,y,sr,chroma):
    import matplotlib.pyplot as plt
    song = np.zeros((chroma.shape[0],chroma.shape[1]))
    print(song.shape)
    print(np.ceil(song.shape[0]/512))
    for onst in tim:
        try:
            ost = float(onst)
            #song[int(sr*ost)] = 1
            print(np.ceil(int(sr*ost)/512)) 
            song[:,np.ceil(int(sr*ost)/512)] = 1 
        except:
            continue  
    plot_matrix(song)

def plot_feat(matrix):
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns; sns.set()
    import pandas as pd

    #matrix = matrix[matrix[:,matrix.shape[1] -1].argsort()]
    data = pd.DataFrame(matrix)
    #data = data.set_index([med_vec])
    sns.heatmap(data)
    labels = ['0','1','2','3','4','5','6','7','8','9','10','11']
    #labels = range(11)
    indexes = np.arange(len(matrix))
    #plt.yticks(indexes+0.5, labels)
    plt.yticks(rotation=0)
    plt.show()


def plot_matrix(matrix):
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns; sns.set()
    import pandas as pd

    #matrix = matrix[matrix[:,matrix.shape[1] -1].argsort()]
    data = pd.DataFrame(matrix)
    #data = data.set_index([med_vec])
    sns.heatmap(data)
    labels = ['6','5','4','3','2','1','0','-1','-2','-3','-4','-5']
    #labels = ['-5','-4','-3','-2','-1','0','1','2','3','4','5','6']
    indexes = np.arange(len(matrix))
    plt.yticks(indexes+0.5, labels)
    plt.yticks(rotation=0)
    plt.show()

def walktree(top, callback, file_type):
    from collections import defaultdict
    #country_dict = dict.fromkeys(user_prof['country'].unique(),country_songs)
    trans_dict = defaultdict(list)
    chroma_dict = {}
    genres = get_genres('genres.csv')

    '''recursively descend the directory tree rooted at top,
       calling the callback function for each regular file'''
    for f in os.listdir(top):
        pathname = os.path.join(top, f)
        mode = os.stat(pathname)[ST_MODE]
        if S_ISDIR(mode):
            # It's a directory, recurse into it
            walktree(pathname, callback, file_type)
        elif S_ISREG(mode):
            # It's a file, call the callback function
            file_name = pathname[pathname.rindex("-")+2:pathname.rindex(".")].lower().replace(" ","_").replace("'","")
            if os.path.exists('../data/CIC/rs200_harmony_clt/rs200_harmony_clt/'+file_name+"_dt.clt"):
                tim = get_data('../data/CIC/rs200_harmony_clt/rs200_harmony_clt/'+file_name+"_dt.clt")
                if int(len(tim[0])) > 2:
                    dict_song = {}
                    dict_song["chords"] = tim[2].tolist()
                    cic,dcf,fid,chroma = callback(pathname,file_type,pathname.replace(f,""),tim[0])
                    song_metadata = genres[genres['ID'] == fid]
                    dict_song["year"] = str(song_metadata['YEAR'].values[0])
                    dict_song["artist"] = song_metadata['ARTIST'].values[0].strip()
                    dict_song["genre"] = song_metadata['GENRE'].values[0].strip()
                    dict_song["title"] = song_metadata['TITLE'].values[0].strip()
                    dict_song["chroma"] = chroma
                    dict_song["CIC"] = cic
                    dict_song["DCF"] = dcf
                    #trans_dict[fid].extend(list(mtx))
                    chroma_dict[str(fid)] = dict_song

                    #print(chroma_dict)
        else:
            # Unknown file type, print a message
            print ('Skipping %s' % pathname)
    #with open("dcf_data.json","w") as file:
    #    file.write(json.dumps(trans_dict))
    with open("data.json","w") as file:
        file.write(json.dumps(chroma_dict))


def visitfile(file,file_type,dir,tim):
    if not file.startswith('.'):
        if file.endswith('.mp3'):
            print(file)
            file_name = file[file.rindex("-")+2:file.rindex(".")].lower().replace(" ","_")
            file_id = int(file[file.rindex("/")+1:file.rindex("/")+4])
            print(file_id)
            trans_matrix = extract_librosa(file,list(tim))
            dcf = calc_din(trans_matrix)
            cic = calc_sic(trans_matrix)
            #plot_matrix(tmatx)
            return cic,dcf,file_id,trans_matrix.tolist()
if __name__ == '__main__':
    walktree(sys.argv[1], visitfile, 'mp3')

#    tim = get_data('seg.txt')
#    print(list(tim[0]))
#    #trans_matrix = extract_librosa('/home/pc/workspace/dados/ismir2018/The Rolling Stone Magazines 500 Greatest Songs Of All Time/008 - The Beatles - Hey Jude.mp3',list(tim[0]))
#    trans_matrix = extract_librosa('1510788992027.mp4',list(tim[0]))
#    print(trans_matrix.shape)
#    calc_din(trans_matrix)  
#    calc_sic(trans_matrix)  


