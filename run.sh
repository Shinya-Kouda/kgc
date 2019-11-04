BERT_DIR=../01_raw/cased_L-12_H-768_A-12
INPUT_DIR=../01_raw/trivial
python run_kg.py \
  --vocab_file=$BERT_DIR/vocab.txt \
  --bert_config_file=$BERT_DIR/bert_config.json \
  --init_checkpoint=$BERT_DIR/bert_model.ckpt \
  --do_train=True \
  --train_file=$INPUT_DIR/trivial-v1.0.json \
  --do_predict=True \
  --predict_file=$INPUT_DIR/dev-v1.1.json \
  --train_batch_size=10 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=50 \
  --output_dir=./output/ \
  --use_tpu=True \
