import pandas as pd
import torch
from torch import nn
from sklearn.feature_extraction.text import CountVectorizer

class Classifier(nn.Module):
    def __init__(self, input_features):
        super(Classifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=10),
            nn.Linear(in_features=10, out_features=10),
            nn.Linear(in_features=10, out_features=1)
        )

    def forward(self, x):
        return self.layer(x)
df = pd.read_csv('text_data.csv')
bow = CountVectorizer(stop_words='english')
X = bow.fit_transform(df['Text'])
feature_names = bow.get_feature_names_out()
input_features = X.shape[1]
model = Classifier(input_features=input_features)
model.load_state_dict(torch.load('model_weights.pth'))


model.eval()

def chat_with_model():
    while True:
        user_input = input("Enter your text (type 'exit' to end): ")
        
        if user_input.lower() == 'exit':
            break

        # Preprocess and transform user input
        user_input_bow = bow.transform([user_input])
        user_input_tensor = torch.tensor(user_input_bow.todense()).float()

        with torch.no_grad():
            model.eval()
            prediction = torch.sigmoid(model(user_input_tensor)).item()

        if prediction >= 0.5:
            print("Positive prediction")
        else:
            print("Negative prediction")

chat_with_model()