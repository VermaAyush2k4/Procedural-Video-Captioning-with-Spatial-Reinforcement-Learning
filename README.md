# Procedural-Video-Captioning-with-Spatial-Reinforcement-Learning

# Overview :

This project integrates a CNN encoder for spatial feature extraction, GNNs for preserving spatial relationships, and an LSTM encoder to capture temporal dependencies across video frames. The captions are optimized using reinforcement learning with policy gradient methods, enhancing their quality by maximizing the expected reward.

# Table of Contents : 
1)Dependencies

2)Dataset Preparation

3)System Architecture

4)Training

5)Evaluation

6)Metrics

## 1) Dependencies :

Install the required Python libraries before running the code:

"pip install tensorflow numpy matplotlib nltk rouge-score opencv-python"

## 2) Dataset :

### A) Descriptions :

a)MSVD dataset over here="https://www.dropbox.com/scl/fo/f4v3thtmsotg53uwd1sz0/AK1HWPBV6iHQyzyGrp2unYA?rlkey=1g9aja3ncdpouybu3u4xyuscp&e=1&dl=0".

b)64639 training data.

c)16160 testing data.

### B) Captions:

a)Each video has a corresponding caption describing its content.

b)Captions are tokenized and padded during preprocessing.

### C) Word Embeddings :

a)A vocabulary (word2idx) is created for mapping words to indices.

b)Unknown words are replaced with a default <unk> token.

## 3) System Architecture :

![cv drawio (1)](https://github.com/user-attachments/assets/7a8514b0-4e4a-4728-9b0e-86d3dddfe0d0)

## 4) Training :

### A) Hyperparameters:

a)batch_size: Number of videos per batch (default: 20).

b)video_frame_steps: Number of frames per video (default: 80).

c)caption_frame_steps: Maximum length of captions (default: 20).

d)dim_hidden: Dimension of LSTM hidden states (default: 1000).

e)epochs: Number of training epochs (default: 50).


### B) Procedure:

a)Load video features and captions in batches.

b)Create masks for valid frames and words.

c)Train the model using TensorFlow.

d)Save model weights every 10 epochs.

### C) Loss Monitoring:

a)Calculate and plot validation loss at the end of each epoch.

b)Loss progression is saved as images in /kaggle/working/loss_imgs.

## 5) Evaluation :

### BLEU Scores:

            a)Compute individual BLEU scores for each generated caption.
            b)Calculate an average BLEU score for the entire corpus.

### ROUGE Scores:

            Evaluate overlap-based similarity (ROUGE-1, ROUGE-2, ROUGE-L) between generated and reference captions.

## 6) Metrices:

The evaluation includes the following metrics:

### BLEU (Bilingual Evaluation Understudy):

            Measures n-gram overlap between generated captions and references.
### ROUGE (Recall-Oriented Understudy for Gisting Evaluation):

            Measures recall-based overlap of unigrams, bigrams, and longest common subsequences.
