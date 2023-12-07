# bash/ppo_tuning.sh -c 0 -b 14res -d data/origin_syn/14res -t ./output/extraction/pseudo_labeled_syn/yelp2023.json -g {your_generator_path} -a {your_alignment_model_path} -f {your_fluency_model_path}

while getopts ':c:b:d:t:g:a:f:' opt
do
    case $opt in
        c)
        CUDA_IDS="$OPTARG" ;;
        b)
        subname="$OPTARG" ;;
        d)
        data_dir="$OPTARG" ;;
        t)
        train_data_dir="$OPTARG" ;;
        g)
        actor_path="$OPTARG" ;;
        a)
        alignment_model_path="$OPTARG" ;;
        f)
        fluency_model_path="$OPTARG" ;;
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
    data_dir="data/origin_syn/14res"
fi


if [ ! "${train_data_dir}" ]
then
    train_data_dir="./output/extraction/pseudo_labeled_syn/yelp2023.json"
fi


seed=42
max_seq_length1=150
max_seq_length2=100
gradient_clip_val=1
weight_decay=1e-6

precision=bf16

max_epochs=2
max_timesteps=1
val_check_interval=1


train_batch_size=3072
eval_batch_size=160
learn_batch_size=40
chunk_size=192
actor_lr=10


# train_batch_size=3072
# eval_batch_size=160
# learn_batch_size=24
# chunk_size=96
# actor_lr=10

output_dir="./output/generation_ppo"



CUDA_VISIBLE_DEVICES=${CUDA_IDS} python ppo_tuning.py \
  --gpus=1 \
  --precision=${precision} \
  --data_dir "${data_dir}" \
  --train_data_dir "${train_data_dir}" \
  --actor_path "${actor_path}" \
  --alignment_model_path "${alignment_model_path}" \
  --fluency_model_path "${fluency_model_path}" \
  --output_dir "${output_dir}" \
  --actor_lr ${actor_lr}e-5 \
  --train_batch_size ${train_batch_size} \
  --eval_batch_size ${eval_batch_size} \
  --learn_batch_size ${learn_batch_size} \
  --chunk_size ${chunk_size} \
  --seed $seed \
  --gradient_clip_val_manual ${gradient_clip_val} \
  --weight_decay ${weight_decay} \
  --max_seq_length1 ${max_seq_length1} \
  --max_seq_length2 ${max_seq_length2} \
  --output_sub_dir ${subname} \
  --train_data_dir ${train_data_dir} \
  --max_epochs ${max_epochs} \
  --max_timesteps ${max_timesteps} \
  --val_check_interval ${val_check_interval} \
  --do_train \
  --scale_reward running
