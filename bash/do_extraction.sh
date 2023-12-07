# bash/do_extraction.sh -c 0 -m {extraction_model_path} -d {input_file_pathj} -o {output_file_path}

while getopts ':d:c:m:o:' opt
do
    case $opt in
        d)
        data_dir="$OPTARG" ;;
        c)
        CUDA_IDS="$OPTARG" ;;
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
    data_dir="data/unlabel/yelp2023/100k_1.json"
fi


if [ ! "${output_dir}" ]
then
    output_dir="./output/extraction/pseudo_labeled/yelp2023.json"
fi



seed=42
precision=bf16
max_seq_length=100
eval_batch_size=400



CUDA_VISIBLE_DEVICES=${CUDA_IDS} python do_extraction.py \
  --accelerator='gpu' \
  --devices=1 \
  --precision=${precision} \
  --data_dir "${data_dir}" \
  --model_name_or_path "${model_name_or_path}" \
  --output_dir "${output_dir}" \
  --eval_batch_size ${eval_batch_size} \
  --seed $seed \
  --max_seq_length ${max_seq_length}