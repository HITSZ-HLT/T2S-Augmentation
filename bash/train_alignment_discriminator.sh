# bash/train_alignment_discriminator.sh -c 0 -d ./output/fluency_dataset_14res/

while getopts ':c:b:d:' opt
do
    case $opt in
        c)
        CUDA_IDS="$OPTARG" ;;
        b)
        subname="$OPTARG" ;;
        d)
        data_dir="$OPTARG" ;;
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



if [ ! "${data_dir}" ]
then
    data_dir="./output/fluency_dataset_14res/"
fi



seed=42
max_seq_length=140
gradient_clip_val=1
warmup_steps=0
weight_decay=0.1

precision=bf16
max_epochs=4
val_check_interval=1000



eval_batch_size=200
train_batch_size=48
model_name_or_path="t5-large"
learning_rate=10

output_dir="./output/alignment_model"



CUDA_VISIBLE_DEVICES=${CUDA_IDS} python train_alignment_discriminator.py \
  --accelerator=gpu \
  --devices=1 \
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
  --val_check_interval ${val_check_interval} \
  --output_sub_dir ${subname} \
  --max_epochs ${max_epochs} \
  --do_train