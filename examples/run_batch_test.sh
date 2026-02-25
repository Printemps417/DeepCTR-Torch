export PYTORCH_ALLOC_CONF=expandable_segments:True

/root/autodl-tmp/myenv/bin/python batchscheduler.py \
    --arrival_rates 2500:2600:10 \
	--p99_target_ms 3000 \
	--duration_s 60 \
	--max_batch_size 256 \
	--max_wait_ms 0.5 \
	--max_seq_len 512 --avg_seq_len 64 \
	--num_sparse 300 --num_seq 150 \
	--user_sparse_count 260 \
	--even_vocab_size 4096 --odd_vocab_size 64 \
	--modes baseline,multislice,emblayerV1,emblayerV2 \
	--out_dir ./batchscheduler_outputs_iobound_4mode_test

# /root/autodl-tmp/myenv/bin/python batchscheduler.py \
# 	--arrival_rates 800,1200,1600,2000,2400 \
# 	--p99_target_ms 400 \
# 	--duration_s 30 \
# 	--max_batch_size 512 \
# 	--max_wait_ms 1.5 \
# 	--max_seq_len 64 --avg_seq_len 4 \
# 	--num_sparse 120 --num_seq 60 \
# 	--user_sparse_count 60 \
# 	--even_vocab_size 4096 --odd_vocab_size 64 \
# 	--modes baseline,multislice,emblayerV1,emblayerV2 \
# 	--out_dir ./batchscheduler_outputs_iobound_tuned
