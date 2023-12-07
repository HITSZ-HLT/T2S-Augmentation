# bash/data_synthesis.sh -c 0 -b 14res_100k -g {your_generator_path} -e {your_extractor_path} -a {your_alignment_model_path} -f {your_fluency_model_path} -n 100000 -d data/origin_syn/14res -r output/extraction/pseudo_labeled/yelp2023.json

while getopts ':c:g:b:s:n:e:a:f:d:r:' opt
do
    case $opt in
        c)
        CUDA_IDS="$OPTARG" ;;
        g)
        generator_path="$OPTARG" ;;
        b)
        subname="$OPTARG" ;;
        s)
        seed="$OPTARG" ;;
        n)
        num_augment_example="$OPTARG" ;;
	e)
	extractor_path="$OPTARG" ;;
	a)
	alignment_model_path="$OPTARG" ;;
	f)
	fluency_model_path="$OPTARG" ;;
	d)
	data_dir="$OPTARG" ;;
	r)
	reference_data_dir="$OPTARG" ;;
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


if [ ! "${num_augment_example}" ]
then
    num_augment_example=10000
fi


if [ ! "${data_dir}" ]
then
    data_dir="data/origin_syn/14res"
fi


if [ ! "${reference_data_dir}" ]
then
    reference_data_dir="output/extraction/pseudo_labeled/yelp2023.json"
fi



max_seq_length=100

precision=bf16
eval_batch_size=100
output_dir="./output/augmentation/${subname}_${seed}.json"


echo ${generator_path}
echo ${extractor_path}
echo ${alignment_model_path}
echo ${fluency_model_path}


CUDA_VISIBLE_DEVICES=${CUDA_IDS} python data_synthesis.py \
  --accelerator=gpu \
  --devices=1 \
  --precision=${precision} \
  --data_dir ${data_dir} \
  --reference_data_dir ${reference_data_dir} \
  --generator_path "${generator_path}" \
  --extractor_path "${extractor_path}" \
  --alignment_model_path "${alignment_model_path}" \
  --fluency_model_path "${fluency_model_path}" \
  --output_dir "${output_dir}" \
  --eval_batch_size ${eval_batch_size} \
  --seed $seed \
  --max_seq_length ${max_seq_length} \
  --num_augment_example ${num_augment_example}

