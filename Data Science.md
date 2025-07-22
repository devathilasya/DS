# &nbsp;                   Data Science

Data Science is like being a detective — but instead of solving crimes, we are solving business problems using data.

Data Science is the field that uses mathematics, statistics, programming, and domain knowledge to extract insights and knowledge from data.



Key components of Data Science-

| Component                | Role                                                                |

| ------------------------ | ------------------------------------------------------------------- |

| \*\*Data Collection\*\*      | Getting data from websites, devices, surveys, etc.                  |

| \*\*Data Cleaning\*\*        | Fixing errors, removing duplicates, handling missing values         |

| \*\*Exploratory Analysis\*\* | Understanding data patterns using statistics and visualizations     |

| \*\*Model Building\*\*       | Using machine learning to predict or classify outcomes              |

| \*\*Evaluation\*\*           | Measuring how good your predictions or models are                   |

| \*\*Deployment\*\*           | Putting the model into use in real-time applications                |

| \*\*Storytelling\*\*         | Explaining results to non-technical people using charts and reports |



Data science helps make smart decisions:

Tech: Recommending YouTube or Netflix videos

Finance: Detecting credit card fraud

Healthcare: Predicting diseases or outcomes

Transport: Optimizing Uber routes

Retail: Predicting what customers will buy



| Step  | Component                  | What It Means                           | Simple Example (Insurance)                      |

| ----  | -------------------------- | --------------------------------------- | ----------------------------------------------- |

| 1️⃣     | \*\*Data Collection\*\*                | Gather raw data                                         | Customer details, claim history, policy type                         |

|  2️⃣     | \*\*Data Cleaning\*\*                   | Fix missing, wrong, or duplicate values           | Fill missing age, remove negative claim amounts                   |

|  3️⃣     | \*\*Data Exploration (EDA)\*\*       | Summarize and visualize data                       | Plot claim amount by age, check fraud count                         |

|  4️⃣     | \*\*Feature Engineering\*\*           | Create new useful variables                         | Create `claim\_per\_member = claim\_amount / family\_members` |

|  5️⃣     | \*\*Data Splitting\*\*                   | Separate training and testing data                 | 80% data to train, 20% to test                                            |

|  6️⃣     | \*\*Model Building\*\*                 | Choose and apply ML algorithm                     | Use Linear Regression or Decision Trees to predict claim amount |

|  7️⃣     | \*\*Model Evaluation\*\*               | Check how well model performs                    | Use R² score, accuracy, confusion matrix                                |

|  8️⃣     | \*\*Model Deployment\*\*             | Make it available to users                            | Build a web tool to estimate claim amount for agents               |

|  9️⃣     | \*\*Monitoring\*\*                       | Keep checking the model after release            | Track if predictions still make sense after 6 months                 |

|  🔁    | \*\*Retraining\*\*             | Use new data to update the model        | Monthly retraining with latest claims              |



#### Data Wrangling-

Data wrangling (also called data cleaning or data preprocessing) is the process of transforming messy, raw data into clean, usable data for analysis or machine learning.

Goal-Get the data into a shape that a computer can understand and use correctly.

Why we use Data Wrangling-

Because real-world data is  messy — it may have:

Missing values (blanks)

Wrong formats (like age as text)

Duplicates

Inconsistent labels (e.g., "Male", "male", "MALE")

Irrelevant columns



| Step                         | What You Do                      | Python Example                      |

| ---------------------------- | -------------------------------- | ----------------------------------- |

| \*\*1. Load Data\*\*             | Read the data file               | `pd.read\_csv()`                     |

| \*\*2. Clean Columns\*\*         | Rename, fix typos                | `df.columns = ...`                  |

| \*\*3. Handle Missing Values\*\* | Fill or remove blanks            | `df.dropna()` or `df.fillna()`      |

| \*\*4. Convert Data Types\*\*    | Change "age" from text to number | `df\['age'] = df\['age'].astype(int)` |

| \*\*5. Remove Duplicates\*\*     | Remove repeated rows             | `df.drop\_duplicates()`              |

| \*\*6. Normalize / Scale\*\*     | Make numbers comparable          | `StandardScaler()`                  |

| \*\*7. Filter/Select Columns\*\* | Keep only useful features        | `df\[\['age', 'income']]`             |



Why it is Important in Data Science-

Bad data → wrong results

ML models need clean, numerical, consistent data

Analysts need clear data to make accurate decisions



## Machine Learning-

Machine Learning is a method 

A computer learns from data, identifies patterns, and makes decisions with minimal human programming.

-> we give the machine lots of past data

-> It learns patterns

-> Then it can predict or decide something new on its own.

Example (insurance)-

-> past data of customers who made fraudulent claims vs genuine claims.

-> feed this into a machine learning model.

-> when a new claim comes in, the machine can predict whether it’s likely to be fraud or not.



Types of Machine Learning:

##### 1.Supervised Learning- 

Supervised learning is when a model learns from labeled data — where each example in the dataset has both:

Input (features) — e.g., age, premium amount

Output (label/target) — e.g., fraud = yes/no



Example-Insurance Fraud Detection

Features (Input)	         Label (Output)

age=54, claim\_amount=9000	fraud\_reported=Yes

age=36, claim\_amount=16000	fraud\_reported=No

age=48, claim\_amount=22000	fraud\_reported=Yes

Types of Supervised Learning

Classification: Output is a category (Yes/No, Class A/B/C)

&nbsp;Used in fraud detection, spam detection, disease diagnosis

Regression: Output is a real number (like price, score)

Used in predicting insurance premium or claim amount



| Use Case                      | Description                                |

| ----------------------------- | ------------------------------------------ |

| \*\*Insurance Fraud Detection\*\* | Input: claim data → Output: fraud (yes/no) |

| \*\*Email Spam Detection\*\*      | Email text → Label: spam or not            |

| \*\*Medical Diagnosis\*\*         | Symptoms → Disease label                   |

| \*\*Credit Scoring\*\*            | Customer data → Loan approved/rejected     |

| \*\*Sales Forecasting\*\*         | Past sales data → Future sales prediction  |



Algorithms:

Linear Regression

Logistic Regression

Decision Trees

Random Forests

Support Vector Machines (SVM)

Neural Networks (for labeled images, text, etc.)



##### Unsupervised Learning-

In Unsupervised Learning, the data has no labels — the model tries to find patterns or structure on its own.

Customer data for insurance-

| Age | Claim Amount | Premium | Tenure |

| --- | ------------ | ------- | ------ |

| 45  | 12000        | 150     | 60     |

| 34  | 8000         | 100     | 24     |

| 67  | 25000        | 200     | 90     |

There is no fraud/not fraud label. But we want to group similar customers:

Group A: High-risk customers

Group B: Medium-risk

Group C: Low-risk

This is clustering, one type of unsupervised learning.

When do we use Unsupervised Learning-

We don’t have labeled data

We want to explore the structure of data

We want to discover hidden patterns

Types of Unsupervised Learning-

| Type                         | Description                                             | Example Use Case                                       |

| ---------------------------- | ------------------------------------------------------- | ------------------------------------------------------ |

| \*\*Clustering\*\*               | Group similar data points together                      | Segment customers in insurance or marketing            |

| \*\*Dimensionality Reduction\*\* | Reduce number of features for analysis or visualization | PCA for visualizing high-dimensional data              |

| \*\*Association Rules\*\*        | Find rules about item co-occurrence                     | Market basket analysis — “People who buy A also buy B” |



Algorithms used for Unsupervised learning-

| Algorithm | Type                     | What it does                        |

| --------- | ------------------------ | ----------------------------------- |

| K-Means   | Clustering               | Groups data into K clusters         |

| DBSCAN    | Clustering               | Groups data based on density        |

| PCA       | Dimensionality Reduction | Projects data into fewer dimensions |



Use cases-

| Domain     | Example                                                                          |

| ---------- | -------------------------------------------------------------------------------- |

| Insurance  | Customer segmentation, fraud pattern detection (unsupervised pre-fraud spotting) |

| Marketing  | Market segmentation based on behavior                                            |

| Healthcare | Grouping patients with similar symptoms                                          |

| E-commerce | Product recommendation without ratings                                           |

| Finance    | Detecting suspicious activity or unknown risk clusters                           |



| Unstructured Data Type | Task (Supervised Learning) | Example                                 |

| ---------------------- | -------------------------- | --------------------------------------- |

| Text (emails)          | Spam or Not Spam           | Gmail spam filter                       |

| Image                  | Object Recognition         | Self-driving cars detecting stop signs  |

| Audio                  | Speech Recognition         | Siri converting voice to text           |

| Video                  | Action Detection           | YouTube detecting inappropriate content |



What we use in Unsupervised Learning:



Structured Data = Numbers, categories, in rows/columns (used in traditional ML, like regression, fraud detection)



Unstructured Data = Text, images, audio, video (used in deep learning, NLP, computer vision)





Structured Data vs Unstructured Data



| Type             | Structured Data                                   | Unstructured Data                 |

| ---------------- | ------------------------------------------------- | --------------------------------- |

| Format           | Organized in rows/columns (tables)                | Free-form, no fixed format        |

| Example          | Age, Salary, Claim Amount                         | Email body, Image, Voice          |

| Easy to Analyze? | Yes (directly with tools like Excel, SQL, pandas) | No (needs NLP, CV, deep learning) |

| File Type        | CSV, Excel, SQL                                   | JPG, MP4, MP3, TXT, PDF           |







Why do we use supervised learning for unstructured Data?

Yes, but with preprocessing.

Supervised learning needs structured data (input features + labeled output), so we first need to convert unstructured data into a structured form.

Convert to features using:

Bag of Words

TF-IDF-Term Frequency – Inverse Document Frequency

It's a math technique to convert text into numbers so machine learning models can work with it.

Or use Word Embeddings (like Word2Vec or BERT)



why cant we choose classification for unsupervised learning?

We can’t use classification for unsupervised learning because classification requires labeled data, and \*\*unsupervised learning deals with unlabeled data.

Classification needs known categories (labels).

Unsupervised learning doesn’t have categories to begin with — we don’t know what to classify into.

It’s like asking: “Predict if this person likes cats or dogs” — but you were never told what “likes cats” or “likes dogs” looks like in the first place.





->we have labels and unstructured data what kind of data is that either is it supervised , unsupervised and semi supervised

When your data has labels but is also unstructured, the type of learning depends on how much labeled data you have and the goal of the task.

Semi-Supervised Learning-

Some labeled, mostly unlabeled

Used when labeling is expensive or time-consuming

Helps models learn from a small amount of labeled data + large unlabeled data

Often used in:

Medical image diagnosis

Sentiment analysis (when only a few reviews are labeled)

➤ If you have unstructured data with a small set of labels → it's semi-supervised.



| Data Type    | Labeled?          | Type of Learning    | Example                                  |

| ------------ | ----------------- | ------------------- | ---------------------------------------- |

| Structured   | Yes               | Supervised          | Predict house prices                     |

| Unstructured | Yes (enough)      | Supervised          | Classify news articles by topic          |

| Unstructured | No                | Unsupervised        | Group documents by topic                 |

| Unstructured | Partially labeled | \*\*Semi-Supervised\*\* | Medical scans with few labeled anomalies |





How do I choose the right model for the data?

Step-1:Understand the problem

| Problem Type        | Description                                   | Model Family                   |

| ------------------- | --------------------------------------------- | ------------------------------ |

| Classification      | Predict categories/labels (Yes/No, A/B/C…)    | Supervised (LogReg, Tree, SVM) |

| Regression          | Predict a number/continuous value             | Supervised (LinearReg, RF)     |

| Clustering          | Group similar items without labels            | Unsupervised (KMeans, DBSCAN)  |

| Dimensionality Red. | Reduce features (PCA, t-SNE)                  | Unsupervised                   |

| Anomaly Detection   | Find outliers/rare patterns                   | Unsupervised / Semi-supervised |

| Recommendation      | Suggest items                                 | Collaborative Filtering        |

| Sequence Prediction | Predict next item in a sequence (stock, text) | Deep Learning (RNN, LSTM)      |



Step-2:Understand the Data

| Data Type   | Characteristics | Common Models             |

| ----------- | --------------- | ------------------------- |

| Tabular     | Rows/columns    | Linear Models, Trees      |

| Text (NLP)  | Words/sentences | NLP Models (TF-IDF, BERT) |

| Image       | Pixels          | CNNs (Convolutional NN)   |

| Time Series | Sequential/time | ARIMA, LSTM, Prophet      |

| Graph       | Nodes/edges     | GNNs (Graph Neural Nets)  |



Step-3:Check Labels

| Labeled?        | Type of Learning |

| --------------- | ---------------- |

| All Labeled     | Supervised       |

| No Labels       | Unsupervised     |

| Few Labels      | Semi-supervised  |

| Learn by reward | Reinforcement    |



Step-4:Use the Roadmap

Is your output a label? (Yes/No)

→ Yes → Supervised Learning

&nbsp;  → Is it category? → Classification

&nbsp;     → Binary? → Logistic Regression / Tree / SVM

&nbsp;     → Multi-class? → Random Forest / XGBoost / NN

&nbsp;  → Is it number? → Regression

&nbsp;     → Simple? → Linear Regression

&nbsp;     → Complex? → Random Forest / XGBoost / NN



→ No → Unsupervised Learning

&nbsp;  → Want to group data? → Clustering

&nbsp;     → KMeans / DBSCAN / Hierarchical

&nbsp;  → Want to reduce features? → PCA / t-SNE

&nbsp;  → Want to detect anomalies? → Isolation Forest / One-Class SVM



STEP 5: Try Multiple Models and Evaluate



| Metric Type      | Examples                  | When to Use              |

| ---------------- | ------------------------- | ------------------------ |

| Accuracy-based   | Accuracy, F1, Precision   | Classification           |

| Error-based      | MSE, RMSE, MAE            | Regression               |

| Silhouette score | Clustering performance    | Unsupervised Clustering  |

| AUC-ROC          | Imbalanced classification | Fraud detection, medical |



\*\*\*\*Choosing a model is not guessing—it's a mix of domain knowledge + data understanding + model testing.

How to find?

Look at data types (text/image/time series?)

Use EDA (Exploratory Data Analysis) to spot patterns

Try baseline models first, then fine-tune

Consider scalability and interpretability \*\*\*\*























































