READ ME

the datafiles cleaned_transcript_rating and raw_transcript_rating consist the combined data of Mark Alfano and research team, 
and the data of the thesis group of Michal Klincewicz. Both files contain:

VIDEO ID (the ID code of the youtube video)
TRANSCRIPT (either raw or cleaned depending on the file)
RATING (the common conspiracy rating (1 for no conspiracy, 2 for conspiracy with falsifiable statements, 3 for conspiracy with self-sealing arguments)
GROUP (which group rated the video (1 for Alfano and team, 2 for thesis group of Klincewicz)

code for the used functions, such as NLP-cleaning of the transcripts is also included in the folder, so adaptations may be made
and stages of the experiment can be reviewed. 

code for the experiment (final-testing.py) is also included in the folder, as well as all important functions:
this code has a few big stages:
1. loading in data and stratify-shuffle it, then cleaning the data
2. Create two keyword lists using my keywordextractor function
3. TF-IDF Vectorization of the dats using scikit learns TfidfVectorizer function.
4. Calculate four bias-matrices based on similarity between each word in the TF-IDF vocab and the keywords in the keywordlist
    (word2vec & keyword list 1, word2vec & Keyword list 2, GloVe & Keyword list 1, and finally GloVe & Keyword list 2)
5. Apply bias matrix vector to baseline TF-IDF vector to create 4 new biased vectors
6. Training en testing with a SVM Classifier


 