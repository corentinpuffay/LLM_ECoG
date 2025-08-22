To start off you need: 

- a .mat file with the high gamma response to a speech stimulus
- the word alignments of your transcripts 
- the .wav file of your stimulus


Change the paths accordingly and run the following steps:

1. Run CreateWordEmbeddingPerTrial: creates embeddings for each trials of your experiment based on your transcripts
2. Run ProcessECoGEmbeddings.py: created a word-level alignement between neural data and LLM embeddings,
saves under AlignedEmbeddings, per trial, layer, and subject, an array of size (#words, dim_embedding)
saves under AlignedNeuralData, per trial, electrode, and subject, an array of size (#words, #lags)
3. Run Ridge Regression 
