# solutions.py

import pyspark
from pyspark.sql import SparkSession
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator as MCE
from pyspark.sql.functions import col, sum as sum_
from pyspark.ml.evaluation import MulticlassClassificationEvaluator



# --------------------- Resilient Distributed Datasets --------------------- #

### Problem 1
def word_count(filename='huck_finn.txt'):
    """
    A function that counts the number of occurrences unique occurrences of each
    word. Sorts the words by count in descending order.
    Parameters:
        filename (str): filename or path to a text file
    Returns:
        word_counts (list): list of (word, count) pairs for the 20 most used words
    """ 
    # initialize SparkSession object
    spark = SparkSession\
            .builder\
            .appName("app_name")\
            .getOrCreate()
    
    # load the file as a PySpark RDD
    data = spark.sparkContext.textFile(filename)
    
    # split each line into words
    words = data.flatMap(lambda line: line.split())

    # count the number of occurrences of each word
    wordCounts = words.map(lambda word: (word, 1)).reduceByKey(lambda x, y: x + y)
    
    # sort the words by count, in descending order
    wordCounts = wordCounts.sortBy(lambda row: row[1], ascending=False)
    solution = wordCounts.collect()[:20] #first 20 most common words
    
    # end SparkSession
    spark.stop()

    # return a list of the (word, count) pairs for the 20 most used words
    return solution
    
    
### Problem 2
def monte_carlo(n=10**5, parts=6):
    """
    Runs a Monte Carlo simulation to estimate the value of pi.
    Parameters:
        n (int): number of sample points per partition
        parts (int): number of partitions
    Returns:
        pi_est (float): estimated value of pi
    """
    # initialize SparkSession object
    spark = SparkSession\
            .builder\
            .appName("app_name")\
            .getOrCreate()
    
    # uniformly sample points in the square [-1, 1]X[-1, 1]
    points = spark.sparkContext.parallelize(np.random.uniform([-1,-1], [1,1], 
                                                              size=(n*parts, 2)), parts)
    # count the # of points that land within the unit circle
    inside_circle = points.map(lambda row: 1 if np.linalg.norm(row, ord=2) <= 1 else 0)
    point_count = inside_circle.reduce(lambda x, y: x + y)

    # end SparkSession
    spark.stop()

    # multiplying the percentage by 4 gives an estimate for the area of the circle
    solution = 4*point_count/(n*parts)

    return solution


# ------------------------------- DataFrames ------------------------------- #

### Problem 3
def titanic_df(filename='titanic.csv'):
    """
    Calculates some statistics from the titanic data.
    
    Returns: the number of women on-board, the number of men on-board,
             the survival rate of women, 
             and the survival rate of men in that order.
    """
    # initialize SparkSession object
    spark = SparkSession\
            .builder\
            .appName("app_name")\
            .getOrCreate()
    
    # load the titanic dataset specifying the schema
    schema = ('survived INT, pclass INT, name STRING, sex STRING, '
              'age FLOAT, sibsp INT, parch INT, fare FLOAT')
    titanic = spark.read.csv('titanic.csv', schema=schema)

    # find the number of women on-board
    women = titanic.filter(titanic.sex == 'female')
    women_count = women.count()
    # find the number of men on-board
    men = titanic.filter(titanic.sex == 'male')
    men_count = men.count()
    # find the survival rate of women
    survived_women = women.filter((women.survived == 1))
    survival_rate_women = survived_women.count()/women.count()

    # the survival rate of men
    survived_men = men.filter((men.survived == 1))
    survival_rate_men = survived_men.count()/men.count()

    # end SparkSession
    spark.stop()

    return (women_count, men_count, survival_rate_women, survival_rate_men)


### Problem 4
def crime_and_income(crimefile='london_crime_by_lsoa.csv',
                     incomefile='london_income_by_borough.csv', major_cat='Robbery'):
    """
    Explores crime by borough and income for the specified major_cat
    Parameters:
        crimefile (str): path to csv file containing crime dataset
        incomefile (str): path to csv file containing income dataset
        major_cat (str): major or general crime category to analyze
    returns:
        (ndarray): borough names sorted by percent months with crime, descending
    """
    # initialize SparkSession object
    spark = SparkSession\
            .builder\
            .appName("app_name")\
            .getOrCreate()
    
    # read & process crime data
    crime = spark.read.csv(crimefile, header=True, inferSchema=True) \
                     .filter("major_category = '{}'".format(major_cat)) \
                     .groupBy("borough") \
                     .sum("value") \
                     .withColumnRenamed("sum(value)", major_cat + "_total_crime")

    # read & process income data
    income = spark.read.csv(incomefile, header=True, inferSchema=True)

    # combine & select relevant columns, & sort (descending)
    income_crime = crime.join(income, "borough") \
                    .select("borough", major_cat + "_total_crime", "median-08-16") \
                    .sort(major_cat + "_total_crime", ascending=False)

    # convert to numpy array
    np_arr = np.array(income_crime.collect())

    # end SparkSession
    spark.stop()

    # plot
    plt.scatter(np_arr[:, 2].astype(float), np_arr[:, 1].astype(float))
    plt.xlabel("Median Income (2008-2016)")
    plt.ylabel("Total Crimes (2008-2016)")
    plt.title(f'Crime ({major_cat}) vs Income')
    plt.tight_layout()
    plt.show()

    return np_arr


### Problem 5
def titanic_classifier(filename='titanic.csv'):
    """
    Implements a classifier model to predict who survived the Titanic.
    Parameters:
        filename (str): path to the dataset
    Returns:
        metrics (tuple): a tuple of metrics gauging the performance of the model
            ('accuracy', 'weightedRecall', 'weightedPrecision')
    """
    # initialize SparkSession object
    spark = SparkSession\
            .builder\
            .appName("app_name")\
            .getOrCreate()
    
    # load the titanic dataset specifying the schema
    schema = ('survived INT, pclass INT, name STRING, sex STRING, '
              'age FLOAT, sibsp INT, parch INT, fare FLOAT')
    titanic = spark.read.csv('titanic.csv', schema=schema)

    # clean & encode data
    sex_binary = StringIndexer(inputCol='sex', outputCol='sex_binary').fit(titanic)
    onehot = OneHotEncoder(inputCols=['pclass'], outputCols=['pclass_onehot'])
    features_col = VectorAssembler(inputCols=['sex_binary', 'pclass_onehot', 'age', 'sibsp', 'parch', 'fare'], outputCol='features')

    # classifier model
    rf_classifier = RandomForestClassifier(labelCol='survived', featuresCol='features')

    # pipeline
    pipeline = Pipeline(stages=[sex_binary, onehot, features_col, rf_classifier])

    # split data & train & predict
    (trainingData, testData) = titanic.randomSplit([0.7, 0.3])
    model = pipeline.fit(trainingData)
    predictions = model.transform(testData)

    # evaluate
    evaluator = MulticlassClassificationEvaluator(labelCol='survived', predictionCol='prediction', metricName='accuracy')
    accuracy = evaluator.evaluate(predictions)
    evaluator.setMetricName('weightedRecall')
    weightedRecall = evaluator.evaluate(predictions)
    evaluator.setMetricName('weightedPrecision')
    weightedPrecision = evaluator.evaluate(predictions)

    # end SparkSession
    spark.stop()

    return (accuracy, weightedRecall, weightedPrecision)
