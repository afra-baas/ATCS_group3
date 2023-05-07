import argparse
import torch
import pytorch_lightning as pl
# from dataset import SentimentDataset
# from model import SentimentClassifier

def predict(model, dataloader, device):
    # Set model to evaluation mode
    model.eval()
    
    # Disable gradient computation for inference
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to the specified device
            batch = {key: value.to(device) for key, value in batch.items()}
            
            # Perform forward pass and get predictions
            logits = model(batch['input_ids'], batch['attention_mask'])
            predictions = torch.argmax(logits, dim=1)
            
            # Print predictions
            print(predictions)
            
if __name__ == '__main__':
    # Define command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                        help='path to trained model')
    parser.add_argument('--prompt', type=str, required=True,
                        help='prompt to use for prediction')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size for inference')
    args = parser.parse_args()
    
    # Load trained model
    model = SentimentClassifier.load_from_checkpoint(args.model_path)
    
    # Initialize dataset and dataloader
    dataset = SentimentDataset(args.prompt)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)
    
    # Determine device to use for inference
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Generate predictions
    predict(model, dataloader, device)