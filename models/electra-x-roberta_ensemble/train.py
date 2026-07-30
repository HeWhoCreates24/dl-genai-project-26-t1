from models.bert_pretrained.train import train_model

def main():
    models = [
        "google/electra-base-discriminator",
        "FacebookAI/roberta-base"
    ]

    for model_name in models:
        print(f"=== Invoking Training Pipeline for Model: {model_name} ===")
        
        # modeular training function
        train_model(
            model_name=model_name,
            epochs=6,
            batch_size=16,
            lr=2e-5,
            max_length=128
        )

if __name__ == "__main__":
    main()