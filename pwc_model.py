import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np 

import os 
dirname = os.path.dirname(__file__)
csv_path = os.path.join(dirname, "Sum of price by product_category_name_english, Year, Quarter, Month and Day.csv")
df = pd.read_csv(csv_path)


##from chatGPT: use a labelEncoder to ensure that data can be decoded
label_encoder = LabelEncoder()

df['product_category_name_english'] = label_encoder.fit_transform(df['product_category_name_english'])

scaler = StandardScaler()
df[['Year', 'Sum of price']] = scaler.fit_transform(df[['Year', 'Sum of price']])

le_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
le_name_reverse_mapping = {v: k for k, v in le_name_mapping.items()}

X = df[['product_category_name_english', 'Year', 'Quarter']].values
y = df['Sum of price'].values.astype(np.float32).reshape(-1, 1)


quarters = {'Qtr 1': 1, 'Qtr 2': 2, 'Qtr 3': 3, 'Qtr 4': 4}
X[:, 2] = [quarters[q] for q in X[:, 2]]


X = X.astype(np.float32) 


X = torch.tensor(X)
y = torch.tensor(y)  


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, 10)  # Input size: 3 (product_category_name_english, Year, Quarter)
        self.fc2 = nn.Linear(10, 1)  # Output size: 1

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    #for every 10th epoch log the loss 
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


model.eval()
with torch.no_grad():
    next_quarter_predictions = model(X_test)


#Find the best performing category in the next quarter
best_performing_category_index = next_quarter_predictions.argmax().item()
#print(next_quarter_predictions.size())
actual_category_encoded = X_test[best_performing_category_index][0].item()

# print(X_test[best_performing_category_index][0].item())
# best_performing_category = label_encoder.inverse_transform(X_test[best_performing_category_index][0].reshape(-1,))

category_performance = {}
for i in range(3697):
    cost = next_quarter_predictions[i][0].item()
    category = le_name_reverse_mapping[int(X_test[i][0].item())]
    if category not in category_performance:
        category_performance[category] = []
    category_performance[category].append(cost)
    #category = le_name_mappling(X_test[i][0].ite

best_avg_cost = -10**8
best_cat = ""
for category in category_performance:
    costs = category_performance[category]
    avg_cost = sum(costs) / len(costs)
    if avg_cost > best_avg_cost:
        best_cat = category
        best_avg_cost = avg_cost

print(f"Projected best performing category in the next quarter:{best_cat}")
