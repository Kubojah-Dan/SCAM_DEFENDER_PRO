import sys
import pickle
import os

def load_model(model_name):
    model_path = os.path.join(os.path.dirname(__file__), f'../app/models/{model_name}.pkl')
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def main():
    if len(sys.argv) < 3:
        print("Usage: python predict.py [model_type: email/message/url] [text]")
        sys.exit(1)
    
    model_type = sys.argv[1]
    text = sys.argv[2]
    model = load_model(f"{model_type}_model")
    prediction = model.predict([text])[0]
    prob = max(model.predict_proba([text])[0])
    print(f"Prediction: {prediction}, Probability: {prob}")

if __name__ == "__main__":
    main()
