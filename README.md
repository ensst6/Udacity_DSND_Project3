# Disaster Response Pipeline Project
Machine learning pipeline project for Udacity Data Scientist Nanodegree

## Background
Data are disaster-related Tweets and text messages collected by the former Figure Eight (now [Appen](https://appen.com/)). The messages have been pre-labeled as disaster-related or not, and with any of 35 categories (e.g., "food", "shelter") they are deemed to belong to.  The original dataset contains 26,386 messages.  

The goals are to:
1. Develop an extract-transform-load (ETL) pipeline to clean the data, prepare it use by a machine learning (ML) algorithm, ans store it in an SQLite database.
2. Develop an ML pipeline to classify the messages as related or not, and into any of the 35 categories.
3. Deploy a web app that allows a user to input a message for classification, and that displays useful visualizations of the data.

## Development
### ETL Pipeline
Exploratory data analysis for developing the ETL pipeline was carried out in the Jupyter notebook `ETL Pipeline Preparation.ipynb`. The final ETL steps are deployed in `etl_pipeline.py`.  
The raw data for both scripts are in `disaster_messages.csv` and `disaster_categories.csv`.  

Key findings from the cleaning & transformation process:  
1. No messages were identified as belonging to the category `child_alone`, so a supervised ML model will never identify any message as belonging to this category. For this reason, it was dropped. This is why there are 35 total categories in the database, instead of the original 36.  
2. 170 duplicate rows were dropped.  
3. While all other categories were binarized, `related` was coded as [0, 1, 2]. Messages categorized as `related = 2` had none of the other categories selected. Examining the individual messages suggested that their relatedness is ambiguous and they may have been meant for later reclassification. Since no guidance on this category was given or obtainable, and since the messages were a small subset (188/26386), these were dropped.
4. The messages are predominantly in/translated to English, but there were a few which hadn't been translated from the original. fastText was used to identify non-English messages, yielding an estimate of  128 and 10 different languages. Trying to develop a model for each language with such a tiny number of examples was deemed infeasible and these were dropped.  
5. Duplication of the `id` field was found in 35 messages. The message content for these was found to be identical, but the category identifications differed. The second of each message appeared to be associated with more categories. It was assumed these messages were later re-classified, or perhaps manually classified. Ony the second entry of each message was retained.
6. The final dataset contains 25,865 English-language messages with 35 associated categories.  

Note that both the notebook and the script will output the same SQLite database. The notebook is hard-coded to output to `TestETL.db` in the same directory, and to drop the `messages` table when starting. The script requires you to pass the path to the database and will also drop `messages` before populating it.   
The `messages` table stores the final (translated) and original message (where available), the genre it was identified as belonging to, and the 35 category fields, binarized to [0, 1]. The primary key is the integer `id` field.  

### ML Pipeline
Preparatory steps for developing the ML pipeline were carried out in the Jupyter notebook `ML Pipeline Preparation.ipynb`. The final ETL steps are deployed in `train_classifier.py`. Either will automatically download the required [NLTK](https://www.nltk.org) files for language processing.  

Initial ML development findings and steps:  
1. Some of the categories are identified with very few messages. Three (`offer`, `tools`, `shops`) are associated with < 1%, and another 15 with < 5% (note the completely empty `child_alone` category has already been dropped). This may lead the classifier to try to maximize accuracy by not associating any message with these categories. Further optimization of the models shown below to better fit these rare categories would be a logical next step for this project.
2. For the text tokenization step, text was converted to lowercase, punctuation and stopwords were removed (the latter with NLTK's "english" corpus), and stemmed with NLTK's WordNetLemmatizer. As explained below, limiting computational complexity turned out to be an important consideration.
3. An additional step using latent semantic analysis (LSA) to reduce the dimensionality of the word vectors was also performed and compared to the model without it.
4. For preliminary purposes, two ML classifiers with static parameters were compared. These were a linear support vector machine (SVM) with stochastic gradient descent (SGD) optimization, and a random forest classifier (RFC). Both were run with and without LSA.  
5. Since there are 35 sets of labels, each model was run through a multi-output classifier. This results in a model for each category. For each category the macro (i.e. unweighted) average precision, recall, and F1 score were used. This was done because of the sparse categories identified above, so that poor results for the sparse `1` category would be weighted equally with the typically more frequent `0` category.
6. Comparison of the classifiers was done on the basis of the average accuracy, precision, recall, and F1 score for all categories.

#### Preliminary Results
The above preliminary analysis of the two classifiers with static parameters yielded the following for the test data (again, these are averages across all categories). MA here is macro-averaged:  
| Classifier | Accuracy | MA precision | MA recall | MA F1 |  
| ---------- | -------- | ------------ | --------- | ----- |  
| Without LSA: | | | | |   
| RFC        | 0.947 | 0.797 | 0.595 | 0.614 |  
| SVM w/SGD  | 0.937 | 0.595 | 0.539 | 0.540 |  
| With LSA: | | | | |  
| RFC        | 0.941 | 0.704 | 0.573 | 0.587 |  
| SVM w/SGD  | 0.941 | 0.595 | 0.565 | 0.570 |  

Overall, the RFC model performed better than SVM+SGD.  
Additionally, the no-LSA RFC model yielded 6 categories in the test set with zero associated messages, while the SVM+SGD yielded 24. In the LSA-augmented models, RFC had 11 categories with no associated messages and SVM-SGD again had 24.

#### Model Optimization
The ML model was optimized with [scikit-learn's](https://scikit-learn.org) GridSearchCV, which conducts an exhaustive search over a matrix of specified model parameters.  
Since the above preliminary models ran in several minutes, I blithely assumed I could do a massive optimization using both classifier with & without LSA, and with multiple options for the text vectorizer, TF-IDF, and the classifier.  
The initial attempt ran in the notebook for three days without resulting. I tried to multithread the grid search, but that failed. I tried running the notebook in the Udacity environment, hoping to use their processing power instead of my anemic laptop's, but that would time out after an hour or so.   Since this is a class with time constraints, I gradually began cutting the size of the problem to get something that would run in a reasonable amount of time (like 24ish hours). For this reason my results are imperfect and further optimization could probably be done.  
What I ended up doing was:  
- Only using RFC as a classifier, since it performed better on the initial tests  
- Not using LSA, as it failed to improve the RFC's performance on the initial tests.
- Splitting my group of four testing parameters into two groups of two
- Using only two cross-validation steps instead of the default five

The first grid search was run on the following sets of parameters:  
```
TfidfTransformer__use_idf': [True, False],
RandomForestClassifier__n_estimators':  [100, 200]
```

This converged very quickly (~2 hrs) with the following results:  
```
Best Parameters: {'multi_clf__estimator__n_estimators': 200, 'tfidf__use_idf': True}
```  

| Accuracy | MA precision | MA recall | MA F1 |  
| -------- | ------------ | --------- | ----- |   
| 0.947 | 0.783 | 0.593 | 0.611 |   

About the same as the initial model.

The second grid search was run on the following sets of parameters:  
```
CountVectorizer__ngram_range: [(1,1),(1,2)],
RandomForestClassifier__max_features': [0.5, "sqrt"]
```
This took about a day to run, with the following results:  
```
Best Parameters: {'multi_clf__estimator__max_features': 0.5, 'vect__ngram_range': (1, 2)}
```

| Accuracy | MA precision | MA recall | MA F1 |  
| -------- | ------------ | --------- | ----- |   
| 0.948 | 0.766 | 0.684 | 0.708 |   

This model shows improved recall, resulting in a higher F1 score.

Again, the above were for the notebook, where I couldn't find a way to multithread the pipeline/classifier training step. Fortunately, I was able to at least multithread the classifier training when running the script from the console. This allowed me to run a grid search using all four sets of parameters in ~24 hrs. The results of this model are:
```
Best Parameters: {'multi_clf__estimator__max_features': 0.5, 'multi_clf__estimator__n_estimators': 200, 'tfidf__use_idf': False, 'vect__ngram_range': (1, 2)}
```  

| Accuracy | MA precision | MA recall | MA F1 |  
| -------- | ------------ | --------- | ----- |   
| 0.947 | 0.772 | 0.672 | 0.697 |   

This model has slightly improved recall at the expense of slighlty less precision, versus the two-parameter model above.

To try to further optimize, I keyed the scoring on F1 instead of on accuracy, since this should help optimize trade-off between precision and recall. The model and results changed slightly, but not much:
```
Best Parameters: {'multi_clf__estimator__max_features': 0.5, 'multi_clf__estimator__n_estimators': 200, 'tfidf__use_idf': True, 'vect__ngram_range': (1, 2)}
```
| Accuracy | MA precision | MA recall | MA F1 |  
| -------- | ------------ | --------- | ----- |   
| 0.947 | 0.764 | 0.676 | 0.699 |   

Note that the script will print the results for all the individual categories, as well as the above averaged data.

### Web App
This was given to us by Udacity largely pre-coded. The default view has an input box for a test message. Below it, visualizations of the test dataset are shown. These include number of messages by genre, and the top and bottom five categories by percent of messages (excluding the overarching category "related"). Typing in a message runs the ML model on it and highlights its estimated categories.

## Deployment  
### Prerequisites
See `requirements.txt`.  
The Jupyter/iPython packages are only required if you want to run the notebooks containing the preliminary exploratory analyses and modeling.  
For the ETL step, you will also need to download the fastText language identification model [ld.176.bin](https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin).

Coded in `Python 3.8.5`.

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model. The database and pickle file names are hard-coded into the web app, so it's suggested to keep them as below.

    - To run the ETL pipeline that cleans data and stores in database:  
        `python3 data/etl_pipeline.py data/disaster_messages.csv data/disaster_categories.csv data/DisResp.db data/lid.176.bin`
    - To run the ML pipeline that trains the classifier and saves the model in a pkl file:  
        `python3 models/train_classifier.py data/DisResp.db models/disaster_model.pkl`

2. Run the following command in the app's directory to run the web app:  
    `python3 app/run.py`

3. Go to http://0.0.0.0:3001/ to view the app. Type in a message to classify. Click the "Disaster Response Project" link at the top to return to home screen.

## History
Created May 17, 2021

## License  
[Licensed](license.md) under the [MIT License](https://spdx.org/licenses/MIT.html).
