import pandas as pd
import numpy as np

# import utility function
from models.bert_pretrained.infer import get_probabilities

def main():
    test_csv_path = 'data/test.csv'
    sample_sub_path = 'data/sample_submission.csv'
    
    test_df = pd.read_csv(test_csv_path)
    sample = pd.read_csv(sample_sub_path)
    option_letters = ['A', 'B', 'C', 'D', 'E']

    print("Starting Ensemble Inference Pipeline...")
    
    elec_probs = get_probabilities(
        model_name="google/electra-base-discriminator", 
        test_df=test_df
    )
    
    rob_probs = get_probabilities(
        model_name="FacebookAI/roberta-base", 
        test_df=test_df
    )

    # soft voting blend
    w_rob = 0.50
    w_elec = 0.50
    print("\nBlending probabilities and ranking answers...")
    blended_probs = (w_elec * elec_probs) + (w_rob * rob_probs)

    # rank options and format submission
    for i in range(len(test_df)):
        ranked_indices = np.argsort(blended_probs[i])[::-1]
        top_3_preds = [option_letters[idx] for idx in ranked_indices[:3]]
        
        sample.loc[i, 'Prediction'] = " ".join(top_3_preds)

    output_path = 'models/electra-x-roberta_ensemble/submission_ensemble.csv'
    sample.to_csv(output_path, index=False)
    print(f"Saved {output_path} successfully!")

if __name__ == "__main__":
    main()