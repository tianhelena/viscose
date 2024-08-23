vocab = model.words
class_of_interest = "__label__positive"  # Replace with your actual label of interest

important_tokens = []
for token in vocab:
    label, probability = model.predict(token)
    if label[0] == class_of_interest:
        important_tokens.append((token, probability[0]))

# Sort tokens by their predictive probability for the class
important_tokens = sorted(important_tokens, key=lambda x: -x[1])

# Display the top predictive tokens for the class
for token, prob in important_tokens[:10]:
    print(f"Token: {token}, Probability: {prob:.4f}")