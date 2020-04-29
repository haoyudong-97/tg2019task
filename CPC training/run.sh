export GLUE_DIR=/home/jacobwang/code/nlp/GLUE-baselines/glue_data

CUDA_VISIBLE_DEVICES=0 python tune.py \
    --task mrpc \
    --mode train \
    --train_cfg config/train_mrpc.json \
    --model_cfg config/bert_base.json \
    --data_file $GLUE_DIR/MRPC/train.tsv \
    --pretrain_file model/bert_model.ckpt \
    --vocab model/vocab.txt \
    --save_dir saved \
    --max_len 128