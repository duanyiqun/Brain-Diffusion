echo "This scirpt construct .pickle files from .mat files from ZuCo dataset."
echo "This script also generates tenary sentiment_labels.json file for ZuCo task1-SR v1.0 and ternary_dataset.json from filtered StanfordSentimentTreebank"
echo "Note: the sentences in ZuCo task1-SR do not overlap with sentences in filtered StanfordSentimentTreebank "
echo "Note: This process can take time, please be patient..."

# python3 ./util/construct_wave_mat_to_pickle_v1.py -t task1-SR -r /projects/CIBCIGroup/00DataUploading/yiqun/bci -o /projects/CIBCIGroup/00DataUploading/yiqun/bci/ZuCo/dewave_sent
# python3 ./util/construct_wave_mat_to_pickle_v1.py -t task2-NR -r /projects/CIBCIGroup/00DataUploading/yiqun/bci -o /projects/CIBCIGroup/00DataUploading/yiqun/bci/ZuCo/dewave_sent
python3 ./braindiffusion/utils/construct_wave_mat_to_pickle_v1.py -t anyname -r /projects/CIBCIGroup/00DataUploading/yiqun/bci -o /projects/CIBCIGroup/00DataUploading/yiqun/bci/ZuCo/dewave_sent
python3 ./braindiffusion/utils/construct_wave_mat_to_pickle_v2.py -t task2-NR-2.0 -r /projects/CIBCIGroup/00DataUploading/yiqun/bci -o /projects/CIBCIGroup/00DataUploading/yiqun/bci/ZuCo/dewave_sent
# python3 ./util/construct_dataset_mat_to_pickle_v2.py

# python3 ./util/get_sentiment_labels.py
# python3 ./util/get_SST_ternary_dataset.py
