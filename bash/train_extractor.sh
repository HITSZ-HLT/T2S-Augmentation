# bash/train_extractor.sh -c 0

while getopts ':d:c:b:s:l:' opt
do
    case $opt in
        d)
        dataset="$OPTARG" ;;
        c)
        CUDA_IDS="$OPTARG" ;;
        b)
        subname="$OPTARG" ;;
        s)
        seed="$OPTARG" ;;
        l)
        learning_rate="$OPTARG" ;;
        ?)
        exit 1;;
    esac
done



if [ ! "${CUDA_IDS}" ]
then
    CUDA_IDS=0
fi



if [ ! "${subname}" ]
then
    subname="test"
fi



if [ ! "${seed}" ]
then
    seed=42
fi


if [ ! "${dataset}" ]
then
    dataset="origin/14res"
fi



if [ ! "${learning_rate}" ]
then
    learning_rate=20
fi



max_seq_length=-1
gradient_clip_val=1
warmup_steps=0
weight_decay=0

precision=bf16
train_batch_size=16
eval_batch_size=64
max_epochs=20

model_name_or_path="t5-base"
data_dir="data"
output_dir="./output/extraction"



CUDA_VISIBLE_DEVICES=${CUDA_IDS} python train_extractor.py \
  --gpus=1 \
  --precision=${precision} \
  --data_dir "${data_dir}" \
  --model_name_or_path "${model_name_or_path}" \
  --output_dir "${output_dir}" \
  --learning_rate ${learning_rate}e-5 \
  --train_batch_size ${train_batch_size} \
  --eval_batch_size ${eval_batch_size} \
  --seed $seed \
  --warmup_steps ${warmup_steps} \
  --gradient_clip_val ${gradient_clip_val} \
  --weight_decay ${weight_decay} \
  --max_seq_length ${max_seq_length} \
  --output_sub_dir ${subname} \
  --dataset $dataset \
  --max_epochs ${max_epochs} \
  --do_train