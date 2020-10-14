# AITA

[r/AmItheAsshole](https://www.reddit.com/r/AmItheAsshole/) is a reddit subreddit. Users on the subreddit submit posts to be judged 
deeming them either not the asshole or your the asshole, abbreviated NTA and YTA respectively (theres actually two more options, but those are obmitted as they 
are rarely used). Using Principle Component Analysis and Linear Discriminant Analysis this project aims to classify posts into either NTA or YTA. 

### Installation 

The requirments for the project are found in the ```requirments.txt``` file. You can use 
```bash
pip3 install requirment==version
``` 
to install the dependencies. 

### Methods and Findings 

The project uses praw to access the reddit API, downloads the 1000 top posts from the subreddit. 
It then catagorizes them into two catagories, YTA and NTA, before taking a random sample of the NTA posts 
to even out the length of the two catagories (n=95). 

It then transforms each document into TF-IDF vectors and then applies principle component analysis with 100 components to transform the document into "topics". Then it 
applies linear discriminant analysis to group the topics into catagories. Finally, it tests a training test against the model and determines it has a **51%** success rate. 
acknowledging the fact a random model will successed **50%** of the time, this model is not effective. 

