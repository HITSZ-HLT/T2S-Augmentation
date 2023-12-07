# bash/train_generator.sh -c 0 -d ./data/origin_syn/14res -t ./output/extraction/pseudo_labeled/yelp2023.json

while getopts ':d:c:b:t:s:' opt
do
    case $opt in
        c)
        CUDA_IDS="$OPTARG" ;;
        d)
        data_dir="$OPTARG" ;;
        t)
        train_data_dir="$OPTARG" ;;
        b)
        subname="$OPTARG" ;;
        s)
        max_steps="$OPTARG" ;;
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



if [ ! "${subname}" ]
then
    subname="test"
fi


if [ ! "${max_steps}" ]
then
    max_steps=10_000
fi



seed=42
max_seq_length1=150
max_seq_length2=100
gradient_clip_val=1
warmup_steps=0
weight_decay=0


# train_batch_size=30
# learning_rate=10

train_batch_size=20
learning_rate=8

# train_batch_size=24
# learning_rate=5

precision=bf16
eval_batch_size=100
val_check_interval=1_000

model_name_or_path=t5-large
output_dir="./output/generation"



CUDA_VISIBLE_DEVICES=${CUDA_IDS} python train_generator.py \
  --gpus=1 \
  --precision=${precision} \
  --data_dir "${data_dir}" \
  --train_data_dir "${train_data_dir}" \
  --model_name_or_path "${model_name_or_path}" \
  --output_dir "${output_dir}" \
  --learning_rate ${learning_rate}e-5 \
  --train_batch_size ${train_batch_size} \
  --eval_batch_size ${eval_batch_size} \
  --seed $seed \
  --warmup_steps ${warmup_steps} \
  --gradient_clip_val ${gradient_clip_val} \
  --weight_decay ${weight_decay} \
  --max_seq_length1 ${max_seq_length1} \
  --max_seq_length2 ${max_seq_length2} \
  --output_sub_dir ${subname} \
  --train_data_dir ${train_data_dir} \
  --max_steps ${max_steps} \
  --val_check_interval ${val_check_interval} \
  --do_train