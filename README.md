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

<img width="160" alt="image" src="https://github.com/user-attachments/assets/4c9e93cd-cf8d-4d68-b421-adb8df60bb91" />


import pandas as pd

from google.colab import files
uploaded = files.upload()
df = pd.read_csv('/content/Encoding Data.csv')

<img width="512" alt="image" src="https://github.com/user-attachments/assets/717daf40-0bbe-4518-b742-30ee5bd8fb44" />


df

<img width="212" alt="image" src="https://github.com/user-attachments/assets/9dd5e67a-5e6f-4d12-8e71-55c2ed971992" />

from sklearn.preprocessing import OrdinalEncoder

temp = ['Hot','Warm','Cold']
oe = OrdinalEncoder(categories=[temp])

oe.fit_transform(df[['ord_2']])

<img width="128" alt="image" src="https://github.com/user-attachments/assets/52b2970b-04aa-48e6-bba5-9da3506d811a" />


df['bo2'] = oe.fit_transform(df[['ord_2']])
df

<img width="268" alt="image" src="https://github.com/user-attachments/assets/4289e949-94e0-41e5-95ea-4cb7b78a38b4" />


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dfc = df.copy()
dfc['ord_2'] = le.fit_transform(dfc['ord_2'])

dfc
<img width="249" alt="image" src="https://github.com/user-attachments/assets/742a7176-bdc2-4e09-b9af-1a89a595dbb0" />

from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

ohe = OneHotEncoder(sparse_output=False)
df2 = df.copy()

enc = pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))

df2 = pd.concat([df,enc],axis=1)

df2
<img width="350" alt="image" src="https://github.com/user-attachments/assets/34e292f0-63ef-4cf3-907a-9ddfd7d92778" />

pd.get_dummies(df2,columns=["nom_0"])

<img width="481" alt="image" src="https://github.com/user-attachments/assets/3d5adcc1-3ae2-4b37-825f-7664ac412e13" />


pip install --upgrade category_encoders

<img width="883" alt="image" src="https://github.com/user-attachments/assets/00bffc0d-5f34-49e2-bb61-bbcb4513e33e" />

from category_encoders import BinaryEncoder

from google.colab import files
uploaded = files.upload()
df = pd.read_csv('/content/data.csv')
<img width="168" alt="image" src="https://github.com/user-attachments/assets/bad706e5-2ad4-4c8e-8183-2ed99a0de131" />

df
<img width="371" alt="image" src="https://github.com/user-attachments/assets/bfae33fb-f89c-46cb-b93c-3536e3d8eefc" />


be = BinaryEncoder()

nd = be.fit_transform(df['Ord_2'])

dfb = pd.concat([df,nd],axis=1)

dfb1 = df.copy()

dfb

<img width="481" alt="image" src="https://github.com/user-attachments/assets/2344957c-3fc0-40ef-8599-d00f59ebf228" />

from category_encoders import TargetEncoder

te = TargetEncoder()

cc = df.copy()

cc
<img width="356" alt="image" src="https://github.com/user-attachments/assets/cd2b4112-d38f-4dae-8410-02615b4ff7b7" />


new = te.fit_transform(X=cc["City"], y=cc["Target"])
cc = pd.concat([cc,new],axis=1)
cc
<img width="443" alt="image" src="https://github.com/user-attachments/assets/0b285692-58ed-4980-8b4d-65051d97f7ab" />

import pandas as pd
from scipy import stats
import numpy as np

from google.colab import files
uploaded = files.upload()

# Get the actual filename from the 'uploaded' dictionary
filename = list(uploaded.keys())[0]

# Read the CSV file using the correct filename
df = pd.read_csv(filename)
<img width="329" alt="image" src="https://github.com/user-attachments/assets/01717580-63bb-42f8-a714-2a4e1ab91b74" />


df
<img width="498" alt="image" src="https://github.com/user-attachments/assets/64d657ca-53ed-4e25-afde-7fdbe2ae0599" />

df.skew()

<img width="224" alt="image" src="https://github.com/user-attachments/assets/c7b24b94-4fab-4df5-a430-cd172ce5780a" />

np.log(df["Highly Positive Skew"])

<img width="182" alt="image" src="https://github.com/user-attachments/assets/ddd0ade6-d128-4ef9-9b85-50b09d3bf640" />

np.reciprocal(df["Moderate Positive Skew"])

<img width="206" alt="image" src="https://github.com/user-attachments/assets/9a3f5113-1c03-41e2-a58c-c9e966966c84" />

np.sqrt(df["Highly Positive Skew"])
<img width="217" alt="image" src="https://github.com/user-attachments/assets/7cbda398-36fb-4bec-8e4c-3785155f0439" />


np.square(df["Highly Positive Skew"])
<img width="207" alt="image" src="https://github.com/user-attachments/assets/2c1958fb-4edb-468d-a2d9-c5f995d97765" />


df["Highly Positive Skew_boxcox"], parameter=stats.boxcox(df["Highly Positive Skew"])
df
<img width="494" alt="image" src="https://github.com/user-attachments/assets/cedb5a54-80d8-4ea4-af7c-eb5dc0aa9c29" />


df["Moderate Negative Skew_yeojohnson"], parameters = stats.yeojohnson(df["Moderate Negative Skew"])
df.skew()

<img width="365" alt="image" src="https://github.com/user-attachments/assets/f74cde21-f58d-4ba1-94b8-d45e892f8ec9" />

df["Highly Negative Skew_yeojohnson"], parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
<img width="311" alt="image" src="https://github.com/user-attachments/assets/5dc70d56-ed6a-4404-a9a1-abb6e1b64434" />

from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew"] = qt.fit_transform(df[["Moderate Negative Skew"]])
df

<img width="890" alt="image" src="https://github.com/user-attachments/assets/83ad5d08-9fb7-4a2f-95b2-93c31afc3ba8" />

import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()


<img width="469" alt="image" src="https://github.com/user-attachments/assets/577c9666-81b3-4ce2-ba88-a5d6f485ace7" />

sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()


<img width="476" alt="image" src="https://github.com/user-attachments/assets/c6b1afad-43ab-4c06-b81f-c46a48d485b4" />

from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"] = qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()

<img width="458" alt="image" src="https://github.com/user-attachments/assets/0fa9c999-a1a9-4ddf-9962-d8ddb4efd930" />


df["Highly Negative Skew_1"]= qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()

<img width="447" alt="image" src="https://github.com/user-attachments/assets/0c2ac512-d4a9-4065-9abd-203dd5f13983" />

sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()

<img width="446" alt="image" src="https://github.com/user-attachments/assets/8e2ade3a-a966-4190-8608-07d2c32c26de" />

from google.colab import files
uploaded = files.upload()

# Get the actual filename from the 'uploaded' dictionary
filename = list(uploaded.keys())[0]

# Read the CSV file using the correct filename
dt = pd.read_csv(filename)

<img width="367" alt="image" src="https://github.com/user-attachments/assets/77bc9aea-c616-4e56-ac1e-39f649a6ad17" />

from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution='normal', n_quantiles=891)

dt["Age_1"]=qt.fit_transform(dt[["Age"]])

sm.qqplot(dt["Age"],line='45')
plt.show()

<img width="458" alt="image" src="https://github.com/user-attachments/assets/1d4f9cdf-64e6-4469-9223-3ccfd5152325" />

sm.qqplot(dt["Age_1"],line='45')
plt.show()

<img width="482" alt="image" src="https://github.com/user-attachments/assets/eadb82eb-ec15-4a18-bb7e-75014e37cbcd" />



       # INCLUDE YOUR CODING AND OUTPUT SCREENSHOTS HERE
# RESULT:
       # INCLUDE YOUR RESULT HERE

       
