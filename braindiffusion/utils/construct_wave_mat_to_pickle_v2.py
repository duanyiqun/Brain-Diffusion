
import os
import numpy as np
import h5py
import data_loading_helpers as dh
from glob import glob
from tqdm import tqdm
import pickle
import argparse
parser = argparse.ArgumentParser(description='Specify task name for converting ZuCo v1.0 Mat file to Pickle')
parser.add_argument('-r', '--root_path', help='root path of the dataset', required=True)
parser.add_argument('-t', '--task_name', help='name of the task in /dataset/ZuCo, choose from {task1-SR,task2-NR,task3-TSR}', required=True)
parser.add_argument('-o', '--output_path', help='output path of the processed dataset')
args = parser.parse_args()

task = "NR"

rootdir = "{}/ZuCo/task2-NR-2.0/Matlab_files/".format(args.root_path)

print('##############################')
print(f'start processing ZuCo task2-NR-2.0...')

dataset_dict = {}

for file in tqdm(os.listdir(rootdir)):
    if file.endswith(task+".mat"):

        file_name = rootdir + file

        print('file name:', file_name)
        subject = file.split("ts")[1].split("_")[0]
        print('subject: ', subject)

        # exclude YMH due to incomplete data because of dyslexia
        if subject != 'YMH':
            print("processing with subject {}".format(subject))
            print("the current dataset already contains {}".format(dataset_dict.keys()))
            assert subject not in dataset_dict
            dataset_dict[subject] = []

            f = h5py.File(file_name,'r')
            # print('keys in f:', list(f.keys()))
            sentence_data = f['sentenceData']
            # print('keys in sentence_data:', list(sentence_data.keys()))
            
            # sent level eeg 
            # mean_t1 = np.squeeze(f[sentence_data['mean_t1'][0][0]][()])
            mean_t1_objs = sentence_data['mean_t1']
            mean_t2_objs = sentence_data['mean_t2']
            mean_a1_objs = sentence_data['mean_a1']
            mean_a2_objs = sentence_data['mean_a2']
            mean_b1_objs = sentence_data['mean_b1']
            mean_b2_objs = sentence_data['mean_b2']
            mean_g1_objs = sentence_data['mean_g1']
            mean_g2_objs = sentence_data['mean_g2']
            
            rawData = sentence_data['rawData']
            contentData = sentence_data['content']
            # print('contentData shape:', contentData.shape, 'dtype:', contentData.dtype)
            omissionR = sentence_data['omissionRate']
            wordData = sentence_data['word']
            # print('wordData shape:', wordData.shape, 'dtype:', wordData.dtype)


            for idx in range(len(rawData)):
                # get sentence string
                obj_reference_content = contentData[idx][0]
                # print('obj_reference_content:', f[obj_reference_content])
                sent_string = dh.load_matlab_string(f[obj_reference_content])
                # print('sentence string:', sent_string)
                
                sent_obj = {'content':sent_string}
                
                # get sentence level EEG
                sent_obj['sentence_level_EEG'] = {
                    'rawData': np.squeeze(f[rawData[idx][0]][()])
                }
                # print(sent_obj)
                sent_obj['word'] = []
                # print(f[wordData[idx][0]])
                # get word level data
                word_data, word_tokens_all, word_tokens_has_fixation, word_tokens_with_mask = dh.extract_word_level_data(f, f[wordData[idx][0]])
                
                if word_data == {}:
                    print(f'missing sent: subj:{subject} content:{sent_string}, append None')
                    dataset_dict[subject].append(None)
                    continue
                elif len(word_tokens_all) == 0:
                    print(f'no word level features: subj:{subject} content:{sent_string}, append None')
                    dataset_dict[subject].append(None)
                    continue

                else:                    
                    for widx in range(len(word_data)):
                        data_dict = word_data[widx]
                        
                        word_obj = {'content':data_dict['content'], 'nFixations': data_dict['nFix']}
                        # word_obj['word_level_EEG'] = {'rawEEG': data_dict["RAW_EEG"], 'fix_positions': data_dict["fixPos"]}
                        word_obj['word_level_EEG'] = None
                        sent_obj['word'].append(word_obj)
                        # print([data.shape for data in data_dict["RAW_EEG"]])
                    sent_obj['word_tokens_has_fixation'] = word_tokens_has_fixation
                    sent_obj['word_tokens_with_mask'] = word_tokens_with_mask
                    sent_obj['word_tokens_all'] = word_tokens_all     
                    
                    # print(sent_obj.keys())
                    # print(len(sent_obj['word']))
                    # print(sent_obj['word'][0])

                    dataset_dict[subject].append(sent_obj)

"""output"""
task_name = args.task_name

if dataset_dict == {}:
    print(f'No mat file found for {task_name}')
    quit()

output_dir = '{}/ZuCo/{}/pickle'.format(args.output_path, task_name)
output_name = f'{task_name}-dataset.pickle'
# with open(os.path.join(output_dir,'task1-SR-dataset.json'), 'w') as out:
#     json.dump(dataset_dict,out,indent = 4)

with open(os.path.join(output_dir,output_name), 'wb') as handle:
    pickle.dump(dataset_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('write to:', os.path.join(output_dir,output_name))

"""sanity check"""
print('subjects:', dataset_dict.keys())
print('num of sent:', len(dataset_dict['YAC']))