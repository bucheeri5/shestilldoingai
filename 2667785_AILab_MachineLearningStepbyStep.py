#!/usr/bin/env python
# coding: utf-8

# https://github.com/bucheeri5/saradoesai.github.io

# # Machine Learning by Example: from Start to End

# # Task 1: README!
# 
# #### The tasks in this notebook should be tackled in discussion with your peer group! 
# 
# **By asking and answering each other questions, you learn much more than doing independent work.** The article [“Embracing Digitalization: Student Learning and New Technologies” (Crittenden, Biel & Lovely 2018)](https://journals.sagepub.com/doi/10.1177/0273475318820895) shows how we learn and retain more information when we explain and discuss it with others.
# 
# Before you tackle a machine learning project you should make sure you keep the bigger picture in your head - this is your human input, skill, and imagination -  no AI can do this for you at the moment. The following sketches the typical steps involved in a machine learning project:
# 
# - Step 1: Frame Your Problem
#     - What is the task? - Who will use it in what environment? What are the risks and impact?
#     - How will you measure performance of your model? Measures sufficient to assess potential risks and impacts?
#     - What are the assumptions? Document and review assumptions for bias. Question everything.
# - Step 2: Get Your Data
#     - Download your data - How will your get your data from where? Permissions and licenses? Suitable and reliable?
#     - Take a quick look at the data structure - how big is it? what fields/attributes are there and how many?
#     - Set aside test data - random split or stratified split? 
# - Step 3: Prepare Your Data
#     - Handling Text/Categorical Data
#     - Scaling and Transformation
#     - Separate the labels from the rest of the attributes
# - Step 4: Select and Train Your Model
#     - train and evaluate on the training set
#     - cross-validation
# - Step 6: Test on Completely New Data
# - Step 7: Publish Your Results! Party! &#x1F389; &#x1F389; &#x1F389;
# 
# In this notebook, we will go through some of the key the steps. Some tasks will involve critical reflection, and others will be about coding. 
# 
# Remember that, **if you are taking more than 30 minutes to do one task without any progress**, you should probably take a break. 
# - Note down what you did and what errors you got in a markdown cell. This will help you understand the recurring errors, you will understand where you left off when you come back to it later, and also help you when you discuss the problem with your peers and with the lab tutors.  
# 
# The code in this notebook is modified from that which was made available by Aurélien Géron and his fabulous book ["Hands-On Machine Learning with Scitkit-Learn, Keras & Tensorflow"](https://eleanor.lib.gla.ac.uk/record=b4094676).

# ## Task 1-1: Before You Start: Prepare Your Computing Environment 
# 
# ### Step 1: First, open Glasgow Anywhere Remote Desktop. Use the Student Desktop. 
# You can use your own machine but it can take more time to set up just so for your course work. The remote desktop, in contrast, has almost every package.
# 
# ### Step 2: Go through the standard approach to opening a notebook. 
# - Open a browser (recommend Chrome incognito mode). 
# - Navigate to the course moodle.
# - Download the notebook linked at `Machine Learning by Example from Start to End` from the course [Resources section](https://moodle.gla.ac.uk/course/view.php?id=39566#section-3). Save it in your course project folder on `One Drive - University of Glasgow`.
# - Start Anaconda Navigator. 
# - Launch Jupyter Notebook. 
# - Navigate to your course project folder. Open up the notebook you downloaded.
# 
# Do not opening a Jupyter Notebook directly – always go through Anaconda Navigator. This allows you to clearly see which version notebook you are opening. The correct version leads to the availability of necessary Python libraries.
# 
# ### Step 3: Prepare to upload material to your GitHub repository. 
# Open a web browser (recommend Google Chrome in incognito mode). Navigate to GitHub and log in. Navigate to your repository for the course AI for the Arts and Humanities (A).
# 

# ## Task 1-2: Checking Your Set Up
# 
# It is important not only to check that you have the correct set of software and packages, but also that the version is the right one. If versions are not compatible with your code then it will throw up errors or unexpected results. This is why you need to make these requirements known to people you share your code with (by, for example, by accompanying your code with a requirements.txt file, as included in your previous lab exrcises).
# 
# ### Python
# 
# Check that your Python has version greater than 3.7 using the following code. This is what the code in the noteboook requires.

# In[2]:


import sys # importing the package sys which lets you talk to your computer system.

assert sys.version_info >= (3, 7) #versions are expressed a pair of numbers (3, 7) which is equivalent to 3.7. 


# In[3]:


import sys # importing the package sys which lets you talk to your computer system.

assert sys.version_info >= (3, 8) #versions are expressed a pair of numbers (3, 8) which is equivalent to 3.8. 


# The `assert` statement throws up an error when the statement following it is not true. If it is true, nothing will be shown. Experiment by replacing the numbers in the round brackets to be much bigger. **A Pair of numbers** like `(3, 7)` in round brackets is a data structure known as a **tuple** in programming lingo. 
# 
# ### Scikit-Learn
# 
# Check that your Scikit-Learn package version is greater than 1.0.1. 
# 
# In this case you will need to import `version` which is part of the `packaging` Python library. This allows you to extract/parse version numbers for Python packages/libraries like `sklearn`.

# In[4]:


from packaging import version #import the package "version"
import sklearn # import scikit-learn

assert version.parse(sklearn.__version__) >= version.parse("1.0.1") 


# ### Fonts Used in Figures
# 
# The following code sets some font sizes to be used with `matplotlib.pyplot` (recall we used matplotlib in previous exercises to display visual information or data). You can set different sizes if you like, but too big and it won't look nice, too little and it will illegible. The code is intended to **prettify** your figures to look nicer.

# In[5]:


import matplotlib.pyplot as plt

plt.rc('font', size=14) #general font size
plt.rc('axes', labelsize=14, titlesize=14) #font size for the titles of x and y axes
plt.rc('legend', fontsize=14) # font size for legends
plt.rc('xtick', labelsize=10) # the font size of labels for intervals marked on the x axis
plt.rc('ytick', labelsize=10) # the font size of labels for intervals marked on the y axis


# ### Creating the Folder for Images 
# 
# The code below creates the directory `images/classification` (if it doesn't already exist) and defines the `save_fig()` function which is used to save the figures you create in matplotlib in high resolution.

# In[6]:


from pathlib import Path

IMAGES_PATH = Path() / "images" / "classification"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# ## Task 1-3: Review Machine Learning
# 
# ### Step 1: Create a markdown cell to demonstrate your own reflection
# - Follow the instructions in Part 1, Task 1, to open your Python notebook. And Create a Markdown cell.
# - In your markdown cell embed an image or link to a diagram illustrating the workflow from data to algorithm to model and data to model to predicted output. 
#     - To embed images in your markdown cell, you can use the syntax `![alt text](image.jpg)` where you replace alt text with a description of your image (keep the square brackets!) and replace image.jpg with the file path and name of your image. 
#     - To include a URL, use the syntax `[title](https://www.example.com)` where you replace title  with your own description, and https://www.example.com  with your own URL. Keep all brackets intact. 
#     - For your reference, you can refer to the [markdown cheat sheet](https://www.markdownguide.org/cheat-sheet/)  - note that HTML codes are also understood by your notebook.
# - Explain in your markdown cell how the examples in Lectures 3 & 4 align with the workflow. For example, what is the data, what was the learning algorithm, what is the model and what did the model output in response to new data?
# - Reflect on the range of ways to explain the workflow and the examples to a wider audience, for example, a museum curator?
# 
# ### Step 2: Discuss and report your reflection with your group
# - Get together with your peer group. Take turns to discuss your reflection above. If you have any difficulties, discuss these also.
# - Note down the results of your discussion in your notebook. In particular, note down anything that help you or others learn the topic. What approach could take in your notebook to engage the wider audience with your machine learning code.
# I've created a cell for you to use already below - double click on the area to start editing.

# ***Markdown Cell***
# 1. [Workflow](https://monkeylearn.com/static/17bfbafbae6490f0a06e7a8773373536/d040e/train3.png) (all rights go to monkeylearn)
# 2. In the process of training an AI model, you start by feeding it a set of examples, like showing a robot pictures of different animals labeled with their names. This labeled data serves as a guide for the model. Then, through a form of reinforcement learning, the AI learns to associate features with the correct labels by repeatedly testing and refining its understanding based on feedback. You evaluate its performance with new data, also called as testing data. If errors occur, you provide additional training examples for the model to reinforce its knowledge. This  cycle continues until the model consistently performs well on the test data, and you can consider it ready, having undergone a form of reinforcement learning to improve its accuracy over time. However, this mainly covers supervised learning. Supervised learning has been the main focus of the course, where it can be as simple as training your AI for a decision tree with yes or no answers and it learns eventually to group things; as demonstrated by the workflow. The flow can be applied to categorizing candidates, such as labelling ones who study doctor who is a yes and non doctor who likers as a no, and then looking at new applications and checking whether the model is consistent.
# 3. Different audiences requires an understanding of their background and skills. Based on that, the explanation can be tailored. The key is to provide the information in an accessible manner. Storytelling could be effective. For example, students are taught by teachers, given knowledge (data), and supervised because the curriculum is already categorized. They are then tested on the data, although the examples are not exactly the same, to see whether they will be able to apply their knowledge based on the data that they learned. Another way would be through interactive demonstrations.
# 4. In my peer groups, we discussed comments and machine learning code. Adding comments to code is very critical, not only for engaging a wider audience but also for keeping track of progress. Because code can vary and still achieve the same end result, comments are a good information management practice. information management practice. Another point that came up during discussions is relating to an audience by using examples that recall the information they may encounter, for example, curators with their collections, teachers with their curriculums and students and so on.

# ## Task 1-4: Framing the Problem
# 
# In this notebook, we will be working with two datasets:
# 
# 1)	Tabular data consisting of information about houses in districts within the US state of California, and, 
# 2)	Image pixel data, each image representing a digit handwritten by high school students and employees of the US Census Bureau. 
# 
# The first of these datasets will be used to **predict median housing prices for a given district**. The results of the prediction will be combined with other data to determine whether it is worth investing in a given district. 
# 
# The second of these datasets will be used to **classify hand written digits**. It was originally developed as a way of sorting out the handwritten US zip codes (similar to UK postcodes) at the post office. 
# 
# 
# ### Step 1: Understand how framing the problem affects data selection
# 
# - The academic article [“Rethinking the field of automatic prediction of court decisions”](https://link.springer.com/article/10.1007/s10506-021-09306-3) by Medvedeva, Wieling & Vols (2023), to appreciate how, depending on the objectives, the characteristics of data and algorithm might differ. 
# - Read the BBC article [“AI facial recognition: Campaigners and MPs call for ban”](https://www.bbc.co.uk/news/technology-67022005) to understand that the same data, depending on its use, can raise concern about AI. We will be discussing prediction court decisions further in Week 7.
# - In view of the above, write down your reflection on the importance of framing your problem precisely in a notebook markdown cell - not only to define the task properly, but to understand how your machine learning model will be used down the road. 
# 
# ### Step 2: How to select your algorithm
# 
# In Lecture 2, we discussed how machine learning can be divided into three types: Supervised, Unsupervised, and Reinforcement. Large part of this course is focused on supervised learning – in particular, in this notebook, we will explore this using examples of regression and classification.
# 
# **To refresh your memory, read this short article from Codecademy** – [“Regression vs Classification”](https://www.codecademy.com/article/regression-vs-classification).
# 
# - Discuss with your peer group whether regression or classification would fit better for predicting median housing prices.
# - Discuss with your peer group whether regression or classification would fit better for handwritten digit recognition.
# - Write down the results of the discussion. In particular, report on what you concluded after the discussion and why.
# 
# ### Step 3: Before Data Collection
# 
# Once your problem is defined (e.g. predicting the median housing price of a district), you will need to collect a new data set appropriate for your task, and/or identify existing data sets that can be used for training your model. 
# 
# - Discuss with your peer group what kind of information about housing in a district you think would help predict the median housing price in the district.
# - Discuss how these decisions might depend on geographical and/or cultural differences and how the information you collect would already bias the data. 
# 
# **Note these down the results of the discussions in Steps 2 & 3 in a markdown cell below.** I have already created a markdown cell for your use - just double click the area to begin editing.

# ***Markdown cell for discussions***
# 
# 1. Framing the problem and stating it helps in communicating what the model aims to achieve for users and ensures focus on the task. Reflecting on the articles above, this is especially important when it comes to assessing risks and aligning the problem with broader objectives and needs. This also helps when it comes to developing the machine learning model and choosing the right algorithm- for example, in cases such as social justice perhaps supervised learning is a better option. Down the road, framing the problem and how the machine learning model is to be used can be assessed in terms of risks and clarifying what the model is not suitable for whether it is an issue of data type or ethical considerations. 
# 
# 2. Regression is appropriate when predicting a continuous numerical value, such as median housing prices. Regression models can predict a variable within a range, making them suitable for continuous data like housing prices. On the other hand, classification is suitable when dealing with discrete categories or labels, such as handwritten digit recognition. The model needs to classify each image of a handwritten digit into one of the distinct digit groups, making it a classification problem.  Choosing the algorithm depends on the specific goals of the task and the nature of the data, which is why it is important to frame the problem. Understanding the data set and the intended outcome is crucial to designing effective and accurate predictive models. 
# 
# 3. A range of factors influence housing data. Access to amenities, public transport, crime rates, integrity of the buildings, average household income, could all help predict median housing. In the context of Scotland, we mentioned the SIMD which can predict the median housing price. A phenomenon that my peer group mentioned is the Waitrose effect, which is prominent in the UK.
# 
# 4. Bias can be introduced if the data used to train the model is not representative of the entire population and geographical area. If data collection is skewed towards a particular region or demographic, the model may not generalize well to other areas or cultural contexts. 
# 

# # Working with Data
# 
# Overviews of machine learning and AI often make it seem as though the largest part of machine learning is in training the algorithm. This is misleading. In fact, [Forbes reported that about 80% of data science is related to data preparation](https://www.forbes.com/sites/gilpress/2016/03/23/data-preparation-most-time-consuming-least-enjoyable-data-science-task-survey-says/). This includes, among other things, data cleaning, re-scaling, and labelling.
# 
# If you also include things like keeping track of what you did and why, storing and backing up data generated as well as the data used at the beginning, and recording the evaluation results, data exploration, interpretation, I would say that **95%** of machine learning is involved in **data management**. This can, in fact, be said to be a substantial part of achieving transparency, a corner stone of addressing the ethical concerns regarding AI and bias, fairness, data protection, explainability etc.
# 
# So, let's get some data!

# # Task 2: Getting Data
# 
# Before anything else, you must get the data! There many ways you can get data. In previous labs, you have already seen that scikit-learn's datasets package has some datasets already available to you. You can also get data from a nuumber of places such as OECD, OpenML, Kaggle, individual repositories in GitHub. 
# 
# In reality, data is everywhere. In fact, artists who make use of AI often make their own datasets: for example, check out [Anna Ridler's shell images](https://annaridler.com/the-shell-record-2021), [Caroline Sinders' feminist dataset](https://carolinesinders.com/feminist-data-set/), and [Refik Anadol's coral images](https://refikanadol.com/works-old/artificial-realities-coral/). While these may be owned by the artists, it can inspire new ways of thinking about data.
# 
# In this task we will look at something a little less artistic! &#x1F609;
# 
# - **Example Tabular Data**: dataset comprising housing prices in California in the the United States. This dataset is available on the GitHub, courtesy of Aurelien Geron. 
# - **Example Image Data**: MNIST dataset comprising images of handwritten digits. Handwritten digit recognition with the MNIST dataset is sometimes called the **"Hello World!" of machine learning**. 
# 
# We will use these datasets to carry out the prediction of 

# ## Task 2-1: Download the Data: Example Tabular Data
# 
# The following code defines **function** called `load_housing_data()`. This function retrieves a compressed file avaialable at `https://github.com/ageron/data/raw/main/housing.tgz` and saves it in a folder `datasets` which is in the same folder as this notebook. This will be created if it does not exist. The code will also extract the contents in the folder `datasets`. 

# In[7]:


from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

def load_housing_data(): #defines a function that loads the housing data available as .tgz file on a github URL
    tarball_path = Path("datasets/housing.tgz") # where you will save your compressed data
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True) #create datasets folder if it does not exist
        url = "https://github.com/ageron/data/raw/main/housing.tgz" # url of where you are getting your data from
        urllib.request.urlretrieve(url, tarball_path) # gets the url content and saves it at location specified by tarball_path
        with tarfile.open(tarball_path) as housing_tarball: # opens saved compressed file as housing_tarball
            housing_tarball.extractall(path="datasets") # extracts the compressed content to datasets folder
    return pd.read_csv(Path("datasets/housing/housing.csv")) #uses panadas to read the csv file from the extracted content

housing = load_housing_data() #runs the function defined above


# ### If you've already downloaded and extracted the compressed file - then the following is all you need

# In[8]:


import pandas as pd
from pathlib import Path

housing = pd.read_csv(Path("datasets/housing/housing.csv"))


# ## Take a Quick Look: housing data

# In[9]:


housing.info()


# The result above tells you how many attributes (e.g. longitude, latitude) characterise the dataset. How many are there? The data type float64 is a numerical data type. So, the table above also tells you how many attributes are not numerical. 

# In[10]:


housing["ocean_proximity"].value_counts() # tells you what values the column for `ocean_proximity` can take


# In[11]:


housing.hist(bins=50, figsize=(12, 8))
save_fig("attribute_histogram_plots")  # extra code
plt.show()


# Finally you can run `housing.describe()` to get a summay of the data set `housing`.

# In[12]:


housing.describe()


# At this point you should stop looking at the data until you have set aside test data. This is to prevent inadvertent bias creeping into the machine learning process.

# ## Task 2-2: Download the Data: Example Image Data
# 
# In contrast to tabular data, image data sets are not always read in using pandas. Technically you can do this (as the line below commented out suggests) but as there are no features human-friendly features (such as, median income etc.) - only pixel information, it does not always help to load it as a pandas dataframe, unless the model requires it to be so. Use the command `type` to see what data type `mnist` is - you can see that it is not a `pandas.core.frame.DataFrame`. Dataframes are not the preferred **data structure** for image data.

# In[13]:


from sklearn.datasets import fetch_openml
import pandas as pd

mnist = fetch_openml('mnist_784', as_frame=False, parser='auto')

#mnist_dataframe = pd.DataFrame(data=mnist.data, columns=mnist.feature_names)


# In[14]:


type(mnist)


# ## Take a Quick Look: MNIST

# The command `mnist.info()` will not work here, to get information about the dataset content, because it is not a pandas dataframe. However, since your dataset is part of the `sklearn.datasets` universe, similar tools as those used in the previous lab exercises apply: for example, the keyword `DECSR` - as demonstrated in the first code cell below. The `print` command can be used in conjunction to get some useful context of the dataset structue and origin. 

# In[15]:


print(mnist.DESCR) #Print the description of the mnist dataset


# ## Task 2-3: Review the data description above with your group.
# 
# What is the size of each image?
# 
# Examine how LeCunn, Cortes, and Burges reorganised the NIST data as MNIST. Note that they remixed the data in two ways to create different a training dataset and test dataset. What they did do? Why do you think they did this? Was it justified? 
# 
# Write down the results in a markdown cell below. I have already created a markdown cell below - just double click to edit the content.

# ***Mark Down cell for critique***
# 
# 1. Each image in the MNIST dataset is size-normalized to fit in a 28x28 pixel box. The original NIST images were size-normalized to a 20x20 pixel box while preserving their aspect ratio. The final images in the MNIST dataset are centered in a 28x28 pixel image by computing the center of mass of the pixels. 
# 2. For the training dataset, they chose 30,000 patterns from SD-3 and 30,000 patterns from SD-1, and 5,000 patterns from each dataset for testing. To address the quality difference between SD-3 and SD-1, writer identities for SD-1 were used to unscramble the data, and characters written by the first 250 writers were placed in the training set, while characters written by the remaining 250 writers were placed in the test set. 
# 3. The data from SD-1 and SD-3 were remixed to address the fact that SD-3 (from Census Bureau employees) is clearer and easier to recognise than SD-1 (from high school students). The mixing of datasets helps to ensure that the results are independent of the training and test sets used.  
# 4. Given the differences in the quality and characteristics of data collected from Census Bureau employees (SD-3) and high-school students (SD-1), the remixing is justified in order to create a more balanced and cleaner dataset. The use of writer identities to unscramble the data, as well as the selection of writers for the training and test sets, all help to create a representative dataset. 

# ### To see a full list of keys other than `DESCR` which is available to this dataset You can use the command `mnist.keys()` to see more keys available - run the code below.

# In[16]:


mnist.keys()


# ## Task 2-4: Identifying the Dimension of Images
# 
# You may recognise some of the keys listed above for `mnist.keys()`. For example, you should have seen the key `data` and `target` already in earlier labs. The former will return the image pixel data, while the `target` key will return the labels (the categories or classification) assigned to each of these images. 
# 
# - Create a code cell below to use these keys in Python, to use the `shape` command to verify the number of images in the dataset and how many features (e.g. pixels) represents the image. Print out the shape and the target categories. 
# 
# I have created a cell for you below with the data and target assigned to the **variables** `images` and `categories`. Add a line to print out the shape of the images and the list a assigned categories. Single click on the area to start editing. 

# In[17]:


# cell for python code 

images = mnist.data
categories = mnist.target

# insert lines below to print the shape of images and to print the categories.
print(images.shape)
print(categories)



# Let's take a look at one of the digits in the dataset - the first item.

# In[18]:


#extra code to visualise the image of digits

import matplotlib.pyplot as plt

## the code below defines a function plot_digit. The initial key work `def` stands for define, followed by function name.
## the function take one argument image_data in a parenthesis. This is followed by a colon. 
## Each line below that will be executed when the function is used. 
## This cell only defines the function. The next cell uses the function.

def plot_digit(image_data): # defines a function so that you need not type all the lines below everytime you view an image
    image = image_data.reshape(28, 28) #reshapes the data into a 28 x 28 image - before it was a string of 784 numbers
    plt.imshow(image, cmap="binary") # show the image in black and white - binary.
    plt.axis("off") # ensures no x and y axes are displayed


# In[19]:


# visualise a selected digit with the following code

some_digit = mnist.data[2]
plot_digit(some_digit)
plt.show()


# # Task 3: Setting Aside the Test Data
# 
# To set aside test data, you need to take shuffled and stratified samples. 
# 
# ## Why Do We Shuffle
# 
# The dataset you are working with could be ordered in a specific way (for example, all the data points in a specific class all at the top). If you select a percentage of 20% from the top, you could get data points in only specific classes. By shuffling we can avoid this. As it happens `sklearn` has a nifty function to allow you to split the data inclusive of splitting. This function is called `train_test_split`. See it in action below, using the housing data.

# In[20]:


from sklearn.model_selection import train_test_split

tratio = 0.2 #to get 20% for testing and 80% for training

train_set, test_set = train_test_split(housing, test_size=tratio, random_state=42) 
## assigning a number to random_state means that everytime you run this you get the same split, unless you change the data.


# ## Why Do We Stratify
# Stratified sampling is used to ensure that the distribution of categories in the sampled data closely matches the distribution in the original dataset. This is particularly useful when dealing with imbalanced datasets, where certain classes may be underrepresented.
# 
# If the dataset is skewed so that it contains more samples of a specific kind more than others, sampling randomly will result in your test data not representing the population you would like to test. An example of the estimated probability of getting a bad sample that does not reflect the actual population is provided below. The US population ratio of females in the census is 51.1%. The following is the probability of getting a sample with less than 48.5% or greater than 53.5% females if you take a random sample withoput stratifying: approximately **10.71%** 

# In[21]:


# extra code – shows another way to estimate the probability of bad sample

import numpy as np

# Set the sample size and the ratio of females in the US Population
sample_size = 1000
ratio_female = 0.511

np.random.seed(42)
# Generate a random sample of size sample_size from a population of size 100_000
# Count the number of females in each sample by summing up the values 
samples = (np.random.rand(100_000, sample_size) < ratio_female).sum(axis=1)

# Calculate the proportion of samples falling outside the range (48.5%, 53.5%)
((samples < 485) | (samples > 535)).mean()


# ## Task 3.1: Stratified Sample: Housing Data
# 
# The following code adds a column to the `housing` data to create bins of data according to interval brackets of median income of districts. This is a first step to creating a stratified sample across different income brackets.

# In[22]:


import numpy as np
import pandas as pd


housing["income_cat"] = pd.cut(housing["median_income"],#Segments and sort data values into bins 
                               #This function useful for going from a continuous variable to a categorical variable
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])


# The following code uses the above bins to implement startified sampling - that is, it will randomly sample 20% (because we set test ratio `tratio` to 0.2) from each income bracket defined above.

# In[23]:


from sklearn.model_selection import train_test_split

tratio = 0.2 #to get 20% for testing and 80% for training

strat_train_set, strat_test_set = train_test_split(housing, test_size=tratio, stratify=housing["income_cat"], random_state=42)


# The code below prints out the proportion of each income category in the stratified test set above.

# In[24]:


strat_test_set["income_cat"].value_counts() / len(strat_test_set) #Prints out in order of the highest proportion first.


# Note the attribute `random_state`. Setting this to a specific number like 42 **keeps the split the same everytime you run the code**. Keep in mind that it will not stay the same if you change the underlying dataset (e.g. adding more). 
# 
# Discuss with your peer group, why a stratified sample based on median income is reasonable. Create a markdown cell below to report on the results of the discussion. I have already create one below, so just double click to edit.

# 1. Median income is a reliable indicator of central tendency: The median is a more robust measure of central tendency than the mean, which can be influenced by extreme values (especially when it comes to areas with a high wealth gap between the residents). It is less influenced by outliers and more accurately represents "typical" household income. We can create subgroups that are more representative of the underlying population by stratifying the sample using median income.  
# 
# 2. Income is the primary consideration in determining housing affordability and to assess a population's financial situation. It is likely to indicate housing affordability because it affects how much finances a household has available for housing expenses. By stratifying the sample based on median income, the sample can reflect the diversity of housing affordability experiences across income groups.  
# 
# 3. Stratifying a sample based on median income is relatively straightforward and is a relatively easier value to obtain. Income categories are simple to understand, which can make the analysis and results easier to interpret and communicate to different audiences. 

# In[25]:


type(mnist.data)


# ## Task 3.2: Setting Aside Test Set for Image Data 
# 
# In the case of `mnist` the data is already cleaned prepared, scaled and ordered, so that the training data is the first 60,000 images, followed by the test data which is the last 10,000 images. So you need not shuffle and stratify nor use `train_test_split`. Instead, you can use the following code to set aside your test dataset. The data type of `mnist.data` is `numpy.ndarray` (you can verify this with the command `type`). 
# - By using a colon and then 60000 in a square bracket after `mnist.data`, you are telling the computer that you want all the items up until the 60000th item (not including the 60000th) in the array `mnist.data`. We assign this to the **variable** `X_train`. 
# - Likewise, the first 60000 categories corresponding the the first 60000 images are assigned to the **variable** `y_train`. 
# - By using a colon after 60000, you are telling the computer you would like all the items from the 60000th onwards.
# 
# It is machine learning convention to use upper case `X` for variable names associated with data and lower case `y` in the variable names associated to labels (or categories/classes).

# In[26]:


#All the data up until the 60000th item will be utilized for training
X_train = mnist.data[:60000]
y_train = mnist.target[:60000]

#Data proceeding the 60000th item will be utilized for testing
X_test = mnist.data[60000:]
y_test = mnist.target[60000:]


# # Task 4: Selecting and Training a Model
# 
# You are finally ready to select and train your model. In the following code, we will use linear **regression for the prediction of district housing prices**, and a **convolutional neural network** for classification of hand written digits. For linear regression, we will use `Scikit-Learn`. For the convolutional neural networks we will use the `tensorflow` library with `keras`. Regardless of the model, the general flow is similar:  
# 
# - Import the model from the relevant library. 
# - Create an untrained model instance. 
# - Fit the model to your training data.
# 
# ## Task 4-1: Housing Data and Linear Regression
# 
# With linear regression, you need data, whose values are continuous - not discrete values such as categories. Note that it is not enough for the values to be numbers, which can also be categories (for example, your place in a queue is a number but is never a fraction like 1.33). The feature `income_cat` is another category expressed as a number. 
# 
# Before doing anything else, let's assign a copy of the stratified training set we created earlier to the variable `housing`. You should always work with copies of data and never look at the test set in case we inadvertently use information in the test to improve the performance (**data snooping bias**).
# 
# To do this use the following code.

# In[27]:


housing = strat_train_set.copy()


# ### Step 1: Checking Correlations: Training Set
# 
# Linear regression in essence works by picking up on correlations between features. So it can be useful to explore the correlations especially between the target value you are trying to predict `median_house_value` and the other features in the dataset, e.g. `median_income`.
# 
# The training set we have is of type `pandas.DataFrame`.  For pandas, dataframes have the function `corr` which calculates the correlations for you. The code is below - first it calculates all the correlations between all the pairs of features and saves it in variable `corr_matrix`. 
# 
# We can take a look at correlations for `median_house_value` by using that feature name in a square bracket (**with quotation marks!**). the last part `sort_values(ascending=False)` sorts the correlation to display it in descending order of correlation (that is, most correlated faetures are listed first).

# In[28]:


corr_matrix = housing.corr(numeric_only=True) # argument is so that it only calculates for numeric value features
corr_matrix["median_house_value"].sort_values(ascending=False)


# ### Step 2: Visualise the Correlations
# 
# Pandas also can visualise these correlations as a graph for you. In the code below, we have selected four features (see the variable with that name), to 4 x 4 grid of graphs.

# In[29]:


from pandas.plotting import scatter_matrix

features = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[features], figsize=(12, 8))
save_fig("scatter_matrix_plot")  

plt.show()


# ### Step 3: Separate the Target Labels from Your Data
# 
# In any machine learning task, you need to provide the data and the target Label separately to the machine learning algorithm. Otherwise, they have no way of knowing which of the features is the target label. In our scenario, the label for the housing data is the `median_house_value`. When your data is in a pandas dataframe format, you can simply 1) drop the column with the label to get the data, and, 2) get the column for the target label, to get the labels.

# In[30]:


housing = strat_train_set.drop("median_house_value", axis=1) ## 1)
housing_labels = strat_train_set["median_house_value"].copy() ## 2)


# ### Step 4: Look for Missing Values in the Data
# 
# When working with tabular data, it is quite common to find that some rows are missing values for some of the columns. If you run the `info` command for dataframes (we've done this in [Task 2-1](#Task-2-1:-Download-the-Data:-Example-Tabular-Data) above!). 
# 
# - Running the code will show the total number of entries. By comparing that number to the number of Non-Null entries for each feature (e.g. `total_bedrooms`) you can see whether there are missing values. 
# - If there are no missing values, these numbers should be the same!
# 
# How many values are missing for the number of `total_bedrooms'? 
# Entries in all the categories are 16512, expect for the total bedrooms, which is 16344. This means that 168 values are missing in the total bedrooms category as it doesnt match the rest of the values. 

# In[31]:


housing.info()


# ### Step 5: Handling Missing Values
# 
# To handle the missing values, you need a code in place to tell the machine what to do if there are missing values. In-depth discussion of handling missing values is beyond the scope of this course, but there are three common ways of handing these:
# 
# - (Option 1) Drop the row with missing value. This causes you to lose data points. In our scenario with the housing data, 168 rows will be removed.
# - (Option 2) Drop the column with missing values. This causes you to lose one of your features.
# - (Option 3) Fill in the missing value with some value such as the median or mean or fixed value that makes sense. This is called **imputing**.
# 
# Depending on which approach you take, the performance of your AI could be different. Also, note that, **with Option 1, you will have to remove the corresponding rows in `housing_labels` before using these in a machine learning task**. Following are codes from each of these approaches. Read the comments included in the code for understanding what each cell is doing.

# In[32]:


# this is the code for Option 1 above. 
housing_option1 = housing.copy() #This makes a copy of the data to variable housing_option1, so that we don't mess up the original data.

housing_option1.dropna(subset=["total_bedrooms"], inplace=True)  # option 1 - dropping the rows where total_bedroom is missing values.

housing_option1.info() #look for missing values after rows have been dropped


# In[33]:


housing_option2 = housing.copy() #This makes a copy of the data to variable housing_option1, so that we don't mess up the original data.

housing_option2.drop("total_bedrooms", axis=1, inplace=True)  # option 2 - dropping the column associated with total_bedrooms

housing_option2.info() # checking for missing values in the new data after column has been dropped


# In[34]:


housing_option3 = housing.copy() #This makes a copy of the data to variable housing_option1, so that we don't mess up the original data.

median = housing["total_bedrooms"].median() # calculating median of the value for total_bedrooms to use in filling missing values
housing_option3["total_bedrooms"].fillna(median, inplace=True)  # option 3 - filling missing values with the median

housing_option3.info()


# #### You can also use `SimpleImputer` from the `sklearn.impute` library to fill missing values with the median

# In[35]:


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median") # initialises the imputer

housing_num = housing.select_dtypes(include=[np.number]) ## includes only numeric features in the data

imputer.fit(housing_num) #calculates the median for each numeric feature so that the imputer can use them

housing_num[:] = imputer.transform(housing_num) # the imputer uses the median to fill the missing values and saves the result in variable X


# ### Step 6: Scaling Your Features
# 
# Machine learning algorithms learn better when similar scales are used across all the features. For example, the numeric range of values for `total_rooms` will be totally different from `median_income`.
# 
# Test this with he **min** and **max** values after running the pandas `describe()` function. Code below.

# In[36]:


housing_num.describe()


# You can see that all the features have very different ranges. Bringing these in alignment is called **feature scaling**. There are a number of ways to scale features. Scikit-Learn provides something called MinMaxScaler which scales the values to fit into a range defined by you. Below, the code is provided for when you are fitting it into the range from -1 to 1. AI algorithms often like the mean to be placed at zero, so best to set a range with zero as the mid point value. 

# In[37]:


from sklearn.preprocessing import MinMaxScaler # get the MinMaxScaler
min_max_scaler = MinMaxScaler(feature_range=(-1, 1)) # setup an instance of a scaler
housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)# use the scaler to transform the data housing_num


# Alternatively, Scikit-Learn also provides a method called StandardScaler. This method tries normalise the distributional characteristics by considering mean and standard deviation for each feature, and normalising the values to have standard deviation 1. But, even without knowing the mathematical details, we can simply employ the tools provided by `sklearn` - example below.

# In[38]:


from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(housing_num)


# In[39]:


housing_num[:]=std_scaler.fit_transform(housing_num)


# ### Step 7: Train a Linear Regression Model
# 
# In the first instance we will use the data resulting from the `SimpleImputer` (with the **median** as a strategy) and use   `StandardScaler` for scaling the features. Before we train the Linear Regression model for predicting median housing prices for districts, we need to also apply the scaling to the target labels (in our case, the known median housing prices). The code is provided below.

# In[40]:


from sklearn.preprocessing import StandardScaler  #We are including it here for completeness sake.

target_scaler = StandardScaler() #instance of Scaler
scaled_labels = target_scaler.fit_transform(housing_labels.to_frame()) #calculate the mean and standard deviation and use it to transform the target labels.


# ### Training Step

# In[41]:


from sklearn.linear_model import LinearRegression #get the library from sklearn.linear model

model = LinearRegression() #get an instance of the untrained model
model.fit(housing_num, scaled_labels)
model.fit(housing[["median_income"]], scaled_labels) #fit it to your data
some_new_data = housing[["median_income"]].iloc[:5]  #pretend this is new data

scaled_predictions = model.predict(some_new_data)
predictions = target_scaler.inverse_transform(scaled_predictions)


# In[42]:


some_new_data = housing_num.iloc[:5] #pretend this is new data
some_new_data = housing[["median_income"]].iloc[:5]  # pretend this is new data

scaled_predictions = model.predict(some_new_data)
predictions = target_scaler.inverse_transform(scaled_predictions)


# In[43]:


print(predictions, housing_labels.iloc[:5])


# In[44]:


# extra code – computes the error ratios discussed in the book
error_ratios = predictions[:5].round(-2) / housing_labels.iloc[:5].values - 1
print(", ".join(['{:.1f}%'.format(100*ratio) for ratio in error_ratios]))


# ### Step 8: Cross Validation
# 
# As mentioned in Lecture 4 - pre-recorded lecture - having one training set and one test set to check performance is limited in producing a robust AI model. What you really want to see is a stable performance across many training sets and test sets. IN the first stance you want to test the model on the training set.
# 
# One way to evaluate your model before testing on the new data (the data you set aside as your test data) is cross validation. This where you split your training data into many pieces, then leave on of the pieces out for testing.The code below does that!

# In[45]:


from sklearn.model_selection import cross_val_score

rmses = -cross_val_score(model, housing_num, scaled_labels,
                              scoring="neg_root_mean_squared_error", cv=10)


# In[46]:


pd.Series(rmses).describe()


# ## Task 4-2: Hand Written Digit Classification
# 
# As mentioned earlier, as an example, for the hand written digits, we will use a specific kind of neural network model called a **Convolutional Neural Network (CNN)** model. In the lectures, we learned about general neural networks but not CNN. If you want to get a feel for CNNs, you can watch the Stat Quest Video [Neural Networks Part 8: Image Classification with Convolutional Neural Networks (CNNs)](https://www.youtube.com/watch?v=HGwBXDKFk9I). If you feel confident to go deeper, Chapters 19 and 22 of Russell and Norvig's Book ["Artificial Intelligence: a modern approach"](https://eleanor.lib.gla.ac.uk/record=b3897063) is an excellent read, not to mention Géron's book ["Hands-On Machine Learning with Scikit-Learn, Keras & Tensorflow"](https://eleanor.lib.gla.ac.uk/record=b4094676).
# 
# For this task, **we will move away from `sklearn` and use `tensorflow` and `keras` instead**. Tensorflow and Keras are popular libraries recognised for their usefulness in building neural networks quickly. Although we already loaded the data from `sklearn`, in the code below, we will get it again from `tensorflow.keras.datasets`. This will allow you to experience getting data from another library and also make it easier to work with the subsequent code because everything will happen with tensorflow. 
# 
# The data is already fairly organised, so, the data cleaning part of the operation can be abbreviated. This is however not a characteristic of image data. It is a characteristic of **curated data** which is not the same as real world messy data 9such as the housing data from earlier). 
# 
# The code for importing the libraries and getting the data has been included below. To get these to work, **you will need to have your environment installed with `tensorflow` and `keras`**. In the first line of the first code cell below, you will notice that `tensorflow` is imported as `tf`. This is a recognised convention in the machine learning community. Adopting this convention makes your code more readable for this community. Once you have imported the library that way, `tf` will be used subsequently instead of `tensorflow`.
# 
# ### Step 1: Get the Data

# In[47]:


import tensorflow as tf

mnist = tf.keras.datasets.mnist.load_data()


# ### Step 2: Review What the Data Looks Like  
# 
# You can review information about what this dataset looks like at the Keras page for the [MNIST digits classification dataset](https://keras.io/api/datasets/mnist/). The page makes it clear that `mnist` above is organised as a data type called **tuple** - something that looks like `(a,b)`. The `a` and `b` are tuples themselves, representing training and test data, respectively. Check first that mnist is a **tuple** with following line of code.

# In[48]:


print(type(mnist))


# The **Keras webpages are useful** for looking up and getting information about wide range of keras commands you might encouter in machine learning programs. 

# ### Step 3: How to Get the Data
# 
# To get the data out of `a` and `b`, run the following code. Read the comment for explanation. 

# In[49]:


(X_train_full, y_train_full), (X_test, y_test) = mnist 
# (X_train_full, y_train_full) is the 'tuple' related to `a` and (X_test, y_test) is the 'tuple' related to `b`.
# X_train_full is the full training data and y_train_full are the corresponding labels 
# - labels indicate what digit the image is of, for example 5 if it is an image of a handwritten 5.


# ### Step 4: Scaling the Pixel Values (the features)
# 
# In dealing with images, there are four main comsiderations that most frequently arise: 
# - 1) input size of the image (height and width in terms of pixels)
# - 2) whether you want to move the pixels so that the image is centered in the middle
# - 3) scaling the value of the pixels to be in a specified range. 
# 
# The neural network we will use will works best with pixel values between 0 and 1. Pixels in a black and white image usually have values between 0 and 255. The code below simply rescales these, dividing by 255. There are other ways of scaling this, similar to when we scaled the feature values of the `housing` data.

# In[50]:


X_train_full = X_train_full / 255.
X_test = X_test / 255.


# ### Step 5: Split the Training Data into Training and Validation Data
# 
# We already have data split into training and test data. The **validation data is split from the training data** and used to evaluate the performance during training. This is **different from test data** which is completely new data not seen during training or fine tuning. 
# 
# Test data is used for the final test before publishing the results. In fact in many competitions, the test data is **withheld behind an application interface** so that contestants cannot engage in **data snooping**. 
# 
# The code below takes the last 5000 images for validation data. The second line does the same for the corresponding labels.

# In[51]:


X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]


# ### Step 5: Increasing Dimension to Include Colour Channels
# 
# An image is usually represented as a (width x height) block of pixels. When presenting your images to the neural network, you need to add an extra dimension to your image representation, to indicate the number of colour channels your images are using. Normally, for a greyscale image this would be 1, while for a RBG colour image it would be 3. 
# 
# All in all you will be submitting something that has shape like `(N, W, H, C)` where `N` is number of images, `W` is the width of any one image, `H` is the height of any one image, and `C` is the number channels (1 for greyscale, 3 for colour). 
# 
# All your images are expected to be the same size as it enters the neural network. 
# 
# The mnist dataset currently has a shape like `(N, W, H)`. Your numpy library allows you to add the required extra dimension. The code below does this.

# In[52]:


import numpy as np # you won't need to run this line if you ran it before in this notebook. But for completeness.

X_train = X_train[..., np.newaxis] #adds a dimension to the image training set - the three dots means keeping everything else the same.
X_valid = X_valid[..., np.newaxis]
X_test = X_test[..., np.newaxis]


# ### Step 6: Build the Neural Network and Fit it to the Data

# In[53]:


tf.keras.backend.clear_session()

tf.random.set_seed(42)
np.random.seed(42)

# Unlike scikit-learn, with tensorflow and keras, the model is built by defining each layer of the neural network.
# Below, everytime tf.keras.layers is called it is building in another layer

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu", kernel_initializer="he_normal"),
    tf.keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="relu", kernel_initializer="he_normal"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", 
              metrics=["accuracy"])

model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))


# In[54]:


model.summary() # not necessary for the machine learning task.


# The summary above is not easy to read initially but it is a presentation of each layer. The numbers at the bottom tell you how many parameters need learning in this model. The visualisation can be useful later when you get more used to neural networks if you should continue on to Semester 2.

# ### Step 7: Train and Evaluate the Model 

# In[55]:


model.evaluate(X_test, y_test)


# ### Comparing with Another Model
# 
# Below you are provided with code for using something called **Stochastic Gradient Decent Classifier**. This model applies the stochastic gradient descent optimiser (cf. the **nadam** optimiser used with the CNN above) with any number of algorithms but by default it applies it to a **Support Vector Machine**. 

# In[64]:


# getting the data again from Scikit-Learn, so that we know the image dimensions fit for the model!

from sklearn.datasets import fetch_openml
import pandas as pd

mnist = fetch_openml('mnist_784', as_frame=False, parser='auto')

# getting the data and the categories for the data
images = mnist.data
categories = mnist.target


# **Normally, we would set aside the test data**. 
# 
# But in this experiement we will abbreviate and use the entire data and evaluate using cross validation, especially since we are not intending, on this occasion, to develop our model with the validation step. **Note that running this might take a while - so be patient!**

# In[65]:


from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score

sgd_clf = SGDClassifier(random_state=42)

#cross validation on training data for fit accuracy

accuracy = cross_val_score(sgd_clf, images, categories, cv=10)

print(accuracy)


# You can see that the accuracies across all the validation runs are far below that of the CNN test results above.

# # Task 5: Reflection
# 
# That's it! You've reviewed the machine learning workflow. Before you go, let's reflect on a few things together to fill in the gaps!
# 
# 
# ## Task 5-1: Reflecting on the Machine Learning Workflow
# 
# 
# Get together with your peer group. For the following tasks, you are expected to write a markdown cell describing the workflow required. You are free to include code, but **no Python code is required**. Discuss the following:
# 
# 1. What would you need to do for your code if:
# 
# - Your were to use your own data (for example, discuss survey data data and photos)?
# - You were changing
#     - Your model?
#     - Your scaling method?
#     - Your approach to handling missing data?
# 2. What is the significance of cross validation?
# 
# ### Further exploration
# 
# In this exercise we only considered numerical data from the housing data - that is we left out the feature `ocean_proximity` which is not numerical. Find out about **One Hot Encoding** from Chapter 2 of the [Hands On Machine Learning book](https://eleanor.lib.gla.ac.uk/record=b4094676). Also watch the video on [Word Embedding and Word2Vec](https://www.youtube.com/watch?v=viZrOnJclY0), to get an intuition for **how textual content is transformed into numerical data**.
# 
# ## Task 5-2: Introducing the Tensorflow Playground
# 
# Before you go, let's play a little bit more with Neural Networks. There is an excellent online resource for this. Go to Tensorflow Playground. A screenshot is provided below should it help to verify that you have navigated to the right place!
# 
# 
# Change your data type to "spiral" by clicking on the picture for spiral data on the lefthand side. 
# - The idea is that the point of orange colour is one class and the ones of blue colour is another class. 
# - As the neural network learns you will see the image on the righthand side change background colour (blue/organge) - the class the neural network thinks the points in those regions belong to.  
# 
# #### Task 5-2-1: Finding small networks that perform well.
# 
# - Play around with the interface to get a feel for where everything is. For example, add more hidden layers (each layer is represented as nodes laid out vertically) and/or add nodes in any layer. Do this together in your group. 
# - Try to come up with the smallest network that will bring the training loss down to 0.2 or less. The traning loss is indicated on the right hand side - right underneath the label **Output**.
# - In a Markdown cell below, describe how many layers with how many nodes you had in your network and how many epochs (indicated on the top lefthand corner) for your best model.
# 
# #### Task 5-2-2: Examine the patterns displayed in the network nodes (see the image above). 
# 
# Discuss in your group and note down in a markdown cell: 
# - what kinds of patterns the neural network might be learning at different layers and nodes. It is difficult to determine this for certain but you can get some intuition by hovering over the nodes in the tensorflow playground.
# 
# Markdown cells have been included below for addressing the discussions in Task 5. This is for your convenience - modify as you see fit.

# **Markdown cell for Task 5**
# 
# My own data: 
# 
# 
# I will collect Music streaming from music streaming platforms like Spotify, Apple Music see which Beatles songs are most popular, how often they're played, and when they're played to make the selection.  Based on that, I will select the songs and incorporate them into a machine learning model using neural networks. I will then scale the data to prevent some features from dominating others, through normalizing the volume levels. Songs that have been extracted with missing parts will be eliminated. Then I will use tensorflow to build neural networks. Cross-validation will be employed to evaluate the performance of the model on the test data. I will use words from songs by The Beatles that I did not use in the training data, and then I will see if they are similar in terms of style. 
# 
# 
# Tensorflow PlayGarden
# 
# I managed to get the data to 0.1 by experimenting through the different functions. I found that sin features helped improve the neural network as they were stripy that went on the spiral mode. It was hard determining at first what is the relationship between the ratio, but my group and I expieremented and found that the first two functions went left/right or up/down and sin had more stripes and intersected with the data more.
# ![Tensorflow Playground](images/tensorflowpg.png) 
# 
# 
# 
# 

# # Summary
# 
# In this notebook you learned about the machine learning pipeline. You reviewed the general workflow from class, and reflected on the workflow in the context of two example cases and data (housing data and minist data). You tried out **Linear Regression** and **Convolutional Neural Net Work**. You also briefly looked at something called a **Support Vector Machine** with **Stochastic Gradient Descent** (not covered in the lectures), comparing the performance for handwritten digit recognition. 
# 
# Any one of these algorithms when looked at in detail, can be quite complex in terms of steps, as seen in the lectures and these labs. However, when using convenient libraries such as `sklearn`, many of them can be implemented in just a few lines. Having said that, where much of the complexity comes in is in preparing the data. And the **data needs more preparing when it is just collected from real world scenarios or sources**.
# 
# **When data is curated** (such as the MNIST data), there is less to clean and prepare. However, if we are to discuss AI and bias, we need to to critically look at decisions made at the data curation stage. Often these decisions are not as transparent as it could be, which compromises our ability to assess the suitability of datasets, algorithms, and interpretation of results.
# 
# We also played with the Tensorflow Playground to enhance our intuition for neural networks. 
