# bash/build_alignment_dataset.sh -c 0 -d ./data/origin_syn/14res -t ./output/extraction/pseudo_labeled_syn/yelp2023.json -m {your_extractor_path} -o ./output/alignment_dataset_14res/

while getopts ':d:t:c:m:o:' opt
do
    case $opt in
        c)
        CUDA_IDS="$OPTARG" ;;
        d)
        data_dir="$OPTARG" ;;
        t)
        train_data_dir="$OPTARG" ;;
        m)
        model_name_or_path="$OPTARG" ;;
        o)
        output_dir="$OPTARG" ;;
        ?)
        exit 1;;
    esac
done



if [ ! "${CUDA_IDS}" ]
then
    CUDA_IDS=0
fi


if [ ! "${data_dir}" ]
then
    data_dir="data/origin_syn/14res"
fi


if [ ! "${train_data_dir}" ]
then
    train_data_dir="./output/extraction/pseudo_labeled_syn/yelp2023.json"
fi



if [ ! "${output_dir}" ]
then
    output_dir="./output/alignment_dataset_14res/"
fi



seed=42
precision=bf16
max_seq_length=100
eval_batch_size=125



CUDA_VISIBLE_DEVICES=${CUDA_IDS} python build_alignment_dataset.py \
  --accelerator=gpu \
  --devices=1 \
  --precision=${precision} \
  --data_dir "${data_dir}" \
  --train_data_dir "${train_data_dir}" \
  --model_name_or_path "${model_name_or_path}" \
  --output_dir "${output_dir}" \
  --eval_batch_size ${eval_batch_size} \
  --seed $seed \
  --max_seq_length ${max_seq_length}