## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd

from google.colab import files
uploaded = files.upload()
df = pd.read_csv('/content/Encoding Data.csv')

df

from sklearn.preprocessing import OrdinalEncoder

temp = ['Hot','Warm','Cold']
oe = OrdinalEncoder(categories=[temp])

oe.fit_transform(df[['ord_2']])

df['bo2'] = oe.fit_transform(df[['ord_2']])
df

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dfc = df.copy()
dfc['ord_2'] = le.fit_transform(dfc['ord_2'])

dfc

from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

ohe = OneHotEncoder(sparse_output=False)
df2 = df.copy()

enc = pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))

df2 = pd.concat([df,enc],axis=1)

df2


pd.get_dummies(df2,columns=["nom_0"])

pip install --upgrade category_encoders

from category_encoders import BinaryEncoder

from google.colab import files
uploaded = files.upload()
df = pd.read_csv('/content/data.csv')

df

be = BinaryEncoder()

nd = be.fit_transform(df['Ord_2'])

dfb = pd.concat([df,nd],axis=1)

dfb1 = df.copy()

dfb

from category_encoders import TargetEncoder

te = TargetEncoder()

cc = df.copy()

new = te.fit_transform(X=cc["City"], y=cc["Target"])
cc = pd.concat([cc,new],axis=1)


       # INCLUDE YOUR CODING AND OUTPUT SCREENSHOTS HERE
# RESULT:
       # INCLUDE YOUR RESULT HERE

       
