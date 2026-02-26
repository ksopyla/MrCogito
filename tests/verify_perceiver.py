import torch
from nn.concept_encoder import ConceptEncoderConfig
from nn.concept_encoder_perceiver import (
    ConceptEncoderForMaskedLMPerceiver,
    ConceptEncoderForSequenceClassificationPerceiver
)

def test_perceiver_mlm():
    print("Testing ConceptEncoderForMaskedLMPerceiver...")
    config = ConceptEncoderConfig(
        vocab_size=100,
        concept_num=10,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_sequence_length=50
    )
    model = ConceptEncoderForMaskedLMPerceiver(config)
    model.eval()
    
    batch_size = 2
    seq_len = 20
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    
    # Forward pass
    output = model(input_ids)
    logits = output.logits
    
    print(f"Logits shape: {logits.shape}")
    assert logits.shape == (batch_size, seq_len, 100)
    print("MLM Forward pass successful!")

def test_perceiver_classification():
    print("\nTesting ConceptEncoderForSequenceClassificationPerceiver...")
    config = ConceptEncoderConfig(
        vocab_size=100,
        concept_num=10,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_sequence_length=50,
        num_labels=3
    )
    model = ConceptEncoderForSequenceClassificationPerceiver(config)
    model.eval()
    
    batch_size = 2
    seq_len = 20
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    
    # Forward pass
    output = model(input_ids)
    logits = output.logits
    
    print(f"Logits shape: {logits.shape}")
    assert logits.shape == (batch_size, 3)
    print("Classification Forward pass successful!")

if __name__ == "__main__":
    test_perceiver_mlm()
    test_perceiver_classification()
