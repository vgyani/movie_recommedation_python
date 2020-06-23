# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np 
import gzip
import os
import urllib.request
import string
import re
import requests
from bs4 import BeautifulSoup
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import matplotlib.pyplot as plt

if not os.path.isfile('group9-projFileInput.csv'):
    if not os.path.isfile('rotten_tomatoes.csv'):
        print('Scraping movie critic reviews from Rotten Tomoatoes...')
        
        def getMovies(criticUrl):
            r = requests.get(criticUrl)
            criticHtml = r.text
            bsyc2 = BeautifulSoup(criticHtml, "lxml")
            temp = bsyc2.select('section[id="criticsReviewsChart_main"] > div > div > ul > li')[2].getText()
            temp2 = [int(s) for s in temp.split() if s.isdigit()]
            numPages = math.ceil(int(temp2[2])/50)
            moveRatingsMap = {}
            movieList = []
            ratingList = []

            for i in range(1,numPages + 1,1):
                criticUrl = criticUrl + '?page=' + str(i)
                r = requests.get(criticUrl)
                criticHtml = r.text
                bsyc2 = BeautifulSoup(criticHtml, "lxml")
                temp2 = bsyc2.select('table[class="table table-striped"] > tr')
                for a in temp2:
                    temp3 = a.select('td[class="center"]')[0].getText()
                    movieList.append(temp3)
                for b in temp2:
                    temp4 = b.select('td > span["class"]')[0].get('class')[2]
                    ratingList.append(temp4)
            
            #print(len(movieList))
            #print(len(ratingList))
            for index in range(len(movieList)):     
                moveRatingsMap[movieList[index]] = ratingList[index]
                
            return moveRatingsMap
    
        criticsNameURLMap = {}
        baseUrl = 'https://www.rottentomatoes.com'
        for alphabet in list(string.ascii_lowercase):
                
            url = 'https://www.rottentomatoes.com/critics/authors?letter=' + alphabet
            r = requests.get(url)
            html = r.text
        
            bsyc = BeautifulSoup(html, "lxml")
            critics_table_list = bsyc.findAll('table',
                                  { "class" : "table table-striped borderless" } )
            for alphabetCritics in critics_table_list:
                alphabetCriticsList = alphabetCritics.findAll('a',{"href" : re.compile(r'/critic/')})
                
            for alphabetCritic in  alphabetCriticsList:
                criticUrl = alphabetCritic.get('href')
                criticUrl = baseUrl + criticUrl + '/movies'
                criticsNameURLMap[alphabetCritic.contents[0]] = criticUrl
        
        # i = 0     
        dataList = []
        dataList.clear()
        for k, v in criticsNameURLMap.items():
        #     print(k, v)
            tempDict = getMovies(v)
            for a1, b1 in tempDict.items():
        #         print(a1,b1)
                a11 = a1[1:len(a1)-7]
                a12 = a1[-5:-1]
                a13 = 1 if b1=='fresh' else 0
                dataList.append([k,a11,a12,a13])
        #     i += 1
        #     print(i)
        #     if (i==10):
        #         break
        
        df = pd.DataFrame(dataList, columns = ['Critic', 'Movie', 'Year', 'Binary'])
        pd.set_option('display.max_rows', len(df))
        df.to_csv('rotten_tomatoes.csv')
        pd.reset_option('display.max_rows')

        print('Movie critic reviews successfully scraped!')
        
    if not os.path.isfile('title.basics.tsv.gz'):
        print('Downloading movie details from IMDB...')
        url = r'https://datasets.imdbws.com/title.basics.tsv.gz'
        urllib.request.urlretrieve(url,'title.basics.tsv.gz')
        print('Movie details successfully downloaded!')
        
    if not os.path.isfile('title.ratings.tsv.gz'):
        print('Downloading movie ratings from IMDB...')
        url = r'https://datasets.imdbws.com/title.ratings.tsv.gz'
        urllib.request.urlretrieve(url,'title.ratings.tsv.gz')
        print('Movie ratings successfuly downloaded!')
        
    print('Merging datasets...')
    
    with gzip.open('title.basics.tsv.gz', "rt", newline='', encoding='utf8') as fileIn:
        imdbMovies = pd.read_csv(
                fileIn,
                delimiter='\t',
                na_values = ['\\N'],
                encoding='utf8',
                quoting=3)
        imdbMovies = imdbMovies[imdbMovies.titleType=='movie']
        imdbMovies = imdbMovies.rename(columns={'primaryTitle':'Movie','startYear':'Year'})
        imdbMovies = imdbMovies.drop(['titleType','originalTitle','isAdult','endYear','runtimeMinutes'],axis=1)
    fileIn.close()
    
    with gzip.open('title.ratings.tsv.gz', "rt", newline='', encoding='utf8') as fileIn:
        imdbMovieRatings = pd.read_csv(
                fileIn,
                delimiter='\t',
                na_values = ['\\N'],
                encoding='utf8',
                quoting=3)
    fileIn.close()
    
    rt = pd.read_csv('rotten_tomatoes.csv')
    
    imdb = pd.merge(imdbMovies,imdbMovieRatings, on='tconst')
    merge = pd.merge(imdb, rt, on=['Movie','Year'])
    merge.to_csv('group9-projFileInput.csv',index=False)
    
    print('Datasets successfully merged!\n')

ds = pd.read_csv('group9-projFileInput.csv')
dsDistinct = ds.drop_duplicates(['Movie','Year'], keep='first')

####user input
dsDistinct.reset_index(drop=True, inplace=True)
print("Welcome to Jasper Movie Recommendation Engine\n")
while(True):
    name = input("Enter a movie name: ")
    searchResult = dsDistinct[dsDistinct.Movie==name.title()]
    if searchResult.empty==True:
        print("####[ERROR]#### Your movie could not be found. Please try again.")
        continue
    if searchResult.Movie.count()==1:
        break
    searchResult = searchResult.sort_values(by='Year')
    searchResult = searchResult.reset_index(drop=True)
    while(True):
        print("\nThe following movies were found with the same name:")
        for index, row in searchResult.iterrows():
            s = '{:2}. {} ({})'.format((index+1), row['Movie'], int(row['Year']))
            print(s)
        try:
            selectedIndex = int(input("Enter an index from the list above to select your movie: "))
            if selectedIndex>0 and selectedIndex<=searchResult.Movie.count():
                print('Your selection: ' + str(searchResult.iloc[selectedIndex-1][1]) + " (" + str(int(searchResult.iloc[selectedIndex-1][2])) + ")")
                searchResult = searchResult.iloc[selectedIndex-1]
                break
            else:
                raise Exception
        except:
            print("\n####[ERROR]#### Invalid selection. Try again")
            continue
    break
        
print("Your movie was found!\n")
print("Running recommendation engine....\n")
#'data' is a df merge between imdb and rt data
#'searchResult' is a df containing one record from 'data' of the movie the user has entered

####cosine similarity analysis
dsSimilarity = ds.head(2500)

tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(dsDistinct['genres'].values.astype('U'))
# x = v.fit_transform(df['Review'].values.astype('U'))

cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

results = {}

for idx, row in dsDistinct.iterrows():
    similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
    similar_items = [(cosine_similarities[idx][i], dsDistinct['tconst'][i]) for i in similar_indices]

    # First item is the item itself, so remove it.
    # Each dictionary entry is like: [(1,2), (3,4)], with each tuple being (score, item_id)
    results[row['tconst']] = similar_items[1:]
    
def item(tconst):
    return dsDistinct.loc[dsDistinct['tconst'] == tconst]['Movie'].tolist()[0].split(' - ')[0]

def recommend(item_id, num):
    print("Recommending movies similar to " + item(item_id) + "...")
    print("{:->130}".format(''))
    recs = results[item_id][:num]
    return pd.DataFrame(recs)

# Just plug in any item id here (1-500), and the number of recommendations you want (1-99)
# You can get a list of valid item IDs by evaluating the variable 'ds', or a few are listed below
# =============================================================================
myMovieId = searchResult.iloc[0][1]
recsdf = recommend(item_id=myMovieId, num=100)
recsdf.rename(columns={0:"Recommended Score", 1:"tconst"}, inplace=True)

aggregatedMovieCriticRating = ds.groupby('tconst')['Recommends'].mean()
movieCriticCounts = ds.groupby('tconst')['Recommends'].count()
aggregatedUserCriticRating = ds.groupby('tconst')['averageRating'].first()
userRatingCounts = ds.groupby('tconst')['numVotes'].first()
 
aggregatedMovieCriticRatingDf = aggregatedMovieCriticRating.to_frame()
movieCriticCountsDf = movieCriticCounts.to_frame()
aggregatedUserCriticRatingDf = aggregatedUserCriticRating.to_frame()
userRatingCounts = userRatingCounts.to_frame()
 
mergedRatings = pd.merge(aggregatedMovieCriticRatingDf,movieCriticCountsDf,how='left', on='tconst')
mergedRatings = pd.merge(mergedRatings, aggregatedUserCriticRatingDf,how='left', on='tconst')
mergedRatings = pd.merge(mergedRatings, userRatingCounts,how='left', on='tconst')
mergedRatings.rename(columns={'Recommends_x':'Average Critic Rating',
                          'Recommends_y':'Number of Critic Ratings',
                           'averageRating':'Average User Rating',
                           'numVotes':'Number of User Ratings'}, 
                  inplace=True)

mergedRatings['Average User Rating'] = mergedRatings['Average User Rating']/10
mergedRatings['Combined Rating'] = (0.7 * (mergedRatings['Average Critic Rating'] ) + 0.3 * ((mergedRatings['Average User Rating'])))
mergedRatings = mergedRatings.sort_values(by=['Combined Rating'], ascending=False)
 
combinedData = pd.merge(recsdf, mergedRatings, how="left", on="tconst")
combinedData["Final Rating"] = combinedData['Recommended Score'] * combinedData['Combined Rating']
combinedData = combinedData.sort_values(by=['Final Rating'], ascending=False)
 
top50Rec = combinedData.head(50)
MovieName = ds[['tconst','Movie']]
MovieName = MovieName.drop_duplicates(subset ="Movie", keep = 'first')
 
top50Rec_Final = pd.merge(top50Rec, MovieName,how = 'left', on = 'tconst' )
top50Rec_Final.shape
 
top50Rec_Final.drop_duplicates(subset='Movie', keep = 'first', inplace=True)
top50Rec_Final.sort_values(by = 'Final Rating', ascending= False)
top50Rec_Final = top50Rec_Final[top50Rec_Final.tconst != myMovieId]
top50Rec_Final = top50Rec_Final.dropna(subset=['Movie'])
top20Rec_Final = top50Rec_Final.head(20)
print("{:>30}{:>25}{:>25}{:>25}{:>25}".format('Movie','Average Critic Rating','Average User Rating','Similarity Score','Recommended Rating'))
print("{:->130}".format(''))
for i, row in enumerate(top20Rec_Final.values):
    print("{:>30}{:>25.2}{:>25.2}{:>25.2}{:>25.2}".format(row[8],row[2],row[4],row[0],row[7]))    

# Functions to generate plots 
def Revenue(movieList):
    results = []
    for movie in movieList:
        budget = None
        grossing = None
        movieCode = movie[0]
        movieName = movie[1]
        url = "https://www.imdb.com/title/"+movieCode + "/"
        review_html = requests.get(url).text
        soup = BeautifulSoup(review_html, 'lxml')
        for h4 in soup.find_all('h4'):
            if "Budget:" in h4:
                budget = h4.next_sibling.strip()
                budget = int(re.sub(r'[^0-9]','',budget))
            if "Cumulative Worldwide Gross:" in h4:
                grossing = h4.next_sibling.strip()
                grossing = int(re.sub(r'[^0-9.]','',grossing))
        if(budget != None and grossing != None):
            results.append([movieCode, movieName, budget, grossing])
    return results


# box office plot
movieList = np.array([top20Rec_Final['tconst'],top20Rec_Final['Movie'],top20Rec_Final['Average User Rating'],top20Rec_Final['Number of User Ratings']])
userRating = np.array(movieList[2]).astype(float)
totalRatings =  np.array(movieList[3]).astype(int)
ratingLabel = np.array(movieList[1])
movieList = movieList.transpose()
results = Revenue(movieList)
arr = np.array(results)
arr = arr.transpose()
label = np.array(arr[1])
budget = np.array(arr[2]).astype(int)
revenue = np.array(arr[3]).astype(int)
ratio = revenue/budget
colors = np.random.rand(len(budget))
index = np.arange(len(ratio))
plt.figure(figsize=(15,10))
plt.barh(index, ratio)
plt.ylabel('Movie', fontsize=12)
plt.xlabel('Worlwide Gross/ Budget', fontsize=12)
plt.yticks(index, label, fontsize=10)
# plt.xticks(index, fontsize=10)
plt.title('Box Office Success', fontsize=20)
plt.show()

# user rating plot
x = ratingLabel
y = userRating
z = totalRatings/200
index = np.arange(len(ratingLabel))
colors = np.random.rand(len(ratingLabel))
plt.clf()
plt.figure(figsize=(15,10))
plt.xticks(index, fontsize=10, rotation=90)
plt.title('User Ratings', fontsize=20)
plt.ylabel('User Ratings', fontsize=10)
plt.scatter(x, y, z, alpha=0.5,c=colors)
plt.show()
