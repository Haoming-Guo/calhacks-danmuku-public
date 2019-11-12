# calhacks-danmuku-public
Authors: Haoming Guo, Haoyuan Liu, Tiantian Cao, Zixian Zang  
Description: This is an open repositary for an undergoing project that analyzes bullet comments and provides insights into audience' reactions to a video. Bullet comments, or "danmuku" in Japanese, are comments indexed to a specific time in a video. By performing sentiment and emotion analysis on these comments, we can analyze viewers' reactions at that moment. The ability to do so potentially provides enormous number of crowdsourced labels to train models on videos.  
Technical Details: This project
1. Scrapes reviews from bilibili, assuming 5 star reviews are positive and 1 star reviews are negative  
2. Train Word2Vec and Bidirectional LSTM on the reviews for sentiment analysis.  
3. Use the model to generate time-specific audience sentiment.  
4. Predict highlights of the video.  
