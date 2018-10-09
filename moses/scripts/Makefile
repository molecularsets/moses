DATA_DIR = ./data
CHECKPOINT_DIR = ./checkpoint
SCRIPTS_DIR = .

MODELS = aae char_rnn junction_tree organ vae

DEVICE = cuda:0

train_$(MODELS): $(DATA_DIR)/train.csv $(CHECKPOINT_DIR)
	python $(SCRIPTS_DIR)/$($@:train_%=%)/train.py --device $(DEVICE) --train_load $(DATA_DIR)/train.csv --model_save $(CHECKPOINT_DIR)/$($@:train_%=%)_model.pt --config_save $(CHECKPOINT_DIR)/$($@:train_%=%)_config.pt --vocab_save $(CHECKPOINT_DIR)/$($@:train_%=%)_vocab.pt

%.csv: $(DATA_DIR) 
	python download_dataset.py --output_dir $(DATA_DIR)

$(DATA_DIR):
	mkdir -p $(DATA_DIR)

$(CHECKPOINT_DIR):
	mkdir -p $(CHECKPOINT_DIR)

clean:
	rm -rf $(DATA_DIR) $(CHECKPOINT_DIR)

.PHONY: clean 
.PHONY: train_$(MODELS)
.PHONY: 